"""
llm_strategy/embedding_demo.py
────────────────────────────────
Minimal demonstration of Arabic legal text embedding using AWS Bedrock
Titan Embed Text v2.

This script shows the full embedding workflow described in PLAN.md:
  1. Take a sample Arabic legal article (simulating output from Stage 3)
  2. Normalise the text using the pipeline's Arabic utilities
  3. Call Bedrock to generate a 1,024-dimensional embedding vector
  4. Demonstrate cosine similarity between semantically related and
     unrelated article pairs to validate embedding quality on Arabic text

This is intentionally a standalone script with no database dependency so
it can be run without a full environment setup to validate Bedrock access
and embedding quality.

Prerequisites:
    pip install boto3 python-dotenv pyarabic
    AWS credentials configured with bedrock:InvokeModel permission

Run:
    python llm_strategy/embedding_demo.py
"""

import json
import math
import sys
from pathlib import Path

# Allow running from any directory within the project
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
from dotenv import load_dotenv

load_dotenv()

from config.settings import settings
from src.utils.arabic_utils import normalize_arabic

# ── Sample Arabic legal articles ─────────────────────────────────────────────
# These are constructed samples representative of UAE federal law language.
# They are NOT reproductions of real documents — they are illustrative examples
# that demonstrate the structure and vocabulary of Arabic legal text.

SAMPLE_ARTICLES = {
    "privacy_obligations": {
        "law": "قانون اتحادي رقم (45) لسنة 2021",
        "article": "المادة الثالثة",
        "chapter": "الباب الأول: أحكام عامة في حماية البيانات الشخصية",
        "text": (
            "يجب على المنشآت التي تتولى معالجة البيانات الشخصية اتخاذ جميع الاحتياطات "
            "اللازمة لضمان سلامة هذه البيانات وسريتها، وعدم الإفصاح عنها أو الاطلاع عليها "
            "أو تداولها إلا في الأحوال التي يجيزها القانون، مع التزام المسؤولية الكاملة عن "
            "أي خرق أو تسريب يطال هذه البيانات."
        ),
    },
    "data_breach_penalties": {
        "law": "قانون اتحادي رقم (45) لسنة 2021",
        "article": "المادة الخامسة عشرة",
        "chapter": "الباب الرابع: العقوبات",
        "text": (
            "يُعاقب بالغرامة التي لا تقل عن مائة ألف درهم ولا تجاوز خمسة ملايين درهم "
            "كل من أفصح أو سرّب أو أتاح للغير بيانات شخصية دون الحصول على موافقة صاحبها، "
            "أو خالف أحكام المادة الثالثة من هذا القانون."
        ),
    },
    "labour_annual_leave": {
        "law": "قانون اتحادي رقم (33) لسنة 2021",
        "article": "المادة التاسعة والعشرون",
        "chapter": "الباب الثالث: الإجازات",
        "text": (
            "يستحق العامل الذي أمضى سنة كاملة في الخدمة إجازة سنوية مدفوعة الأجر لا تقل "
            "عن ثلاثين يومًا عن كل سنة خدمة، وتُحسب الإجازة بنسبة ما قضاه في الخدمة إذا "
            "كانت مدة عمله تقل عن سنة."
        ),
    },
    "corporate_governance": {
        "law": "قانون اتحادي رقم (32) لسنة 2021",
        "article": "المادة الثانية والأربعون",
        "chapter": "الباب الثاني: الشركات المساهمة العامة",
        "text": (
            "يتولى مجلس الإدارة الإشراف العام على أعمال الشركة ورسم سياستها واتخاذ القرارات "
            "اللازمة لتحقيق أغراضها، وله في سبيل ذلك التفويض لأحد أعضائه أو لمدير تنفيذي "
            "بممارسة بعض صلاحياته وفق ما تحدده اللائحة التنظيمية."
        ),
    },
    "unrelated_traffic_law": {
        "law": "قانون اتحادي رقم (21) لسنة 1995",
        "article": "المادة السادسة والعشرون",
        "chapter": "الباب الثاني: قواعد المرور",
        "text": (
            "يلتزم قائد المركبة بالتوقف التام عند الإشارة الحمراء، وعدم تجاوز الخطوط "
            "الفاصلة المرسومة على الطريق، ويُحظر عليه استخدام الهاتف المحمول أثناء القيادة "
            "إلا بواسطة السماعة الحرة."
        ),
    },
}


# ── Core embedding functions ──────────────────────────────────────────────────

def build_embedding_input(article: dict) -> str:
    """
    Construct the embedding input string for an article node.

    Following the strategy in PLAN.md: we prepend the chapter heading to the
    article heading and text.  This context is crucial — an article about
    "penalties" (عقوبات) in a data privacy law chapter means something very
    different from the same word in a traffic law chapter.  The embedding
    needs to capture this contextual meaning.
    """
    return f"{article['chapter']} | {article['article']}\n{article['text']}"


def get_embedding(bedrock_client, text: str) -> list[float]:
    """
    Call Bedrock Titan Embed Text v2 and return the embedding vector.

    The Titan v2 model accepts a JSON payload with a single 'inputText' field
    and returns a JSON response with an 'embedding' field containing a list
    of 1,024 floats (dimensions).

    Note: Titan v2 supports input up to 8,192 tokens, which is sufficient for
    even the longest UAE legal articles.
    """
    payload = {"inputText": text}

    response = bedrock_client.invoke_model(
        modelId=settings.bedrock_embed_model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload),
    )

    body = json.loads(response["body"].read())
    return body["embedding"]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Cosine similarity measures the angle between two vectors in high-dimensional
    space, ranging from -1 (opposite) to 1 (identical).  For embedding comparisons
    it is preferable to Euclidean distance because it is insensitive to vector
    magnitude — two articles can both be "very data-privacy-ish" even if one is
    longer than the other, and cosine correctly rates them as similar.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


# ── Main demonstration ────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 70)
    print("ThakAI — Arabic Legal Document Embedding Demonstration")
    print("Model:", settings.bedrock_embed_model_id)
    print("=" * 70)

    # Initialise the Bedrock client
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        **(
            {
                "aws_access_key_id": settings.aws_access_key_id,
                "aws_secret_access_key": settings.aws_secret_access_key,
            }
            if settings.aws_access_key_id
            else {}
        ),
    )

    # Step 1: Normalise all articles and generate embeddings
    print("\n[Step 1] Normalising Arabic text and generating embeddings...")
    embeddings: dict[str, list[float]] = {}

    for name, article in SAMPLE_ARTICLES.items():
        # Build the embedding input string
        raw_input = build_embedding_input(article)

        # Normalise using the same pipeline utilities as ingestion
        # This ensures query embeddings and document embeddings use the same
        # surface form — essential for reliable cosine similarity.
        normalised_input = normalize_arabic(raw_input)

        print(f"\n  Article: [{name}]")
        print(f"  Law:     {article['law']}")
        print(f"  Article: {article['article']}")
        print(f"  Chars before normalisation: {len(raw_input)}")
        print(f"  Chars after normalisation:  {len(normalised_input)}")

        # Call Bedrock
        vector = get_embedding(bedrock, normalised_input)
        embeddings[name] = vector
        print(f"  Embedding: {len(vector)} dimensions, first 5 values: "
              f"{[round(v, 4) for v in vector[:5]]}")

    # Step 2: Query simulation
    # This is the key test: given a natural-language query about data privacy,
    # do we retrieve the data privacy articles over unrelated ones?
    print("\n" + "=" * 70)
    print("[Step 2] Query simulation — 'ما هي عقوبات تسريب البيانات الشخصية؟'")
    print("         (What are the penalties for personal data breaches?)")
    print("=" * 70)

    query = "ما هي عقوبات تسريب البيانات الشخصية؟"
    normalised_query = normalize_arabic(query)
    query_vector = get_embedding(bedrock, normalised_query)
    print(f"\n  Query vector: {len(query_vector)} dims")

    # Compute similarity of the query against all articles
    print("\n  Cosine similarities (higher = more relevant):")
    similarities = []
    for name, vector in embeddings.items():
        sim = cosine_similarity(query_vector, vector)
        article = SAMPLE_ARTICLES[name]
        similarities.append((sim, name, article["law"], article["article"]))

    similarities.sort(reverse=True)
    for rank, (sim, name, law, art_number) in enumerate(similarities, 1):
        marker = " ◄ TOP RESULT" if rank == 1 else ""
        print(f"  #{rank}  {sim:.4f}  [{name}]  {law}, {art_number}{marker}")

    # Step 3: Demonstrate inter-article similarity within the same law
    print("\n" + "=" * 70)
    print("[Step 3] Within-law article similarity")
    print("         (privacy_obligations vs data_breach_penalties — same law,")
    print("          different chapters — should be highly similar)")
    print("=" * 70)

    sim_same_law = cosine_similarity(
        embeddings["privacy_obligations"],
        embeddings["data_breach_penalties"]
    )
    sim_diff_domain = cosine_similarity(
        embeddings["privacy_obligations"],
        embeddings["unrelated_traffic_law"]
    )
    sim_labour_corporate = cosine_similarity(
        embeddings["labour_annual_leave"],
        embeddings["corporate_governance"]
    )

    print(f"\n  privacy_obligations  ↔  data_breach_penalties: {sim_same_law:.4f}")
    print(f"  privacy_obligations  ↔  unrelated_traffic_law: {sim_diff_domain:.4f}")
    print(f"  labour_annual_leave  ↔  corporate_governance:  {sim_labour_corporate:.4f}")

    print("\n  Expected: same-domain pairs score HIGHER than cross-domain pairs.")
    if sim_same_law > sim_diff_domain:
        print("  ✓ PASS: Data privacy articles are more similar to each other "
              "than to the unrelated traffic law article.")
    else:
        print("  ✗ FAIL: Consider switching to a stronger multilingual model "
              "(e.g. Cohere Embed Multilingual v3 via Bedrock).")

    print("\n" + "=" * 70)
    print("Demo complete.  In production, vectors are stored in OpenSearch")
    print("Serverless (1,024-dim k-NN index) and queried with pre-filters on")
    print("jurisdiction, status, and effective_date.  See PLAN.md §5.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
