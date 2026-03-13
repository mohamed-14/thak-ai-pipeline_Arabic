# ThakAI — LLM Application Strategy for Arabic Legal Document Intelligence

**Document status:** Approved design plan  
**Scope:** Post-ingestion LLM integration for Arabic legal documents produced by the v2 ETL pipeline  
**Constraint:** AWS-native services only (Bedrock, OpenSearch, RDS, S3)

---

## 1. Problem Statement

The v2 ingestion pipeline produces a structured, normalised representation of Arabic legal documents: clean text, a document hierarchy (chapters → articles → clauses), and rich metadata. The question this document addresses is: **what do we do with that structured content?**

Legal professionals and compliance analysts need to answer questions like:

- "What are the penalties under Federal Law No. 3 of 2022 for data privacy violations?"
- "Which articles in the UAE labour law were amended between 2020 and 2024?"
- "Summarise all obligations imposed on financial institutions by Cabinet Resolution 50 of 2021."
- "Find all laws that reference Article 14 of the Civil Transactions Law."

These are fundamentally retrieval + reasoning tasks. They cannot be solved by keyword search alone (Arabic morphology is too complex), and they cannot be solved by sending an entire corpus to an LLM in a single prompt (cost and context window limits). The solution is a **Retrieval-Augmented Generation (RAG)** architecture with Arabic-aware embeddings.

---

## 2. Core Insight: Why Arabic Requires Special Attention

Arabic is morphologically rich in a way that directly affects every part of the LLM pipeline. A single Arabic root like ك-ت-ب can produce dozens of surface forms: كتب، يكتب، كاتب، مكتوب، كتابة، مكتبة. This means that a clause about "writing obligations" might use the word كتابة in one law and مكتوبًا in another — they share the same root but look completely different on the surface.

This has two practical consequences for us. First, **embedding quality is critical**: we need a multilingual or Arabic-specific embedding model that encodes semantic meaning rather than surface forms. A model that was trained primarily on English and lightly fine-tuned on Arabic will cluster كتب and مكتوب apart, leading to missed retrievals. Second, **chunk design matters**: because articles in legal documents are logically self-contained and short, we have a natural chunking unit — the article node from Stage 3. We do not need to apply arbitrary character-count chunking, which is a common source of quality degradation in RAG systems.

---

## 3. Recommended Architecture: Hierarchical RAG

The architecture operates at two levels of granularity, which mirrors the natural structure of law:

**Level 1 — Document-level retrieval** answers questions like "which laws are relevant to this topic?" It embeds the full document metadata (title, subject area, summary of the preamble) and returns candidate documents.

**Level 2 — Article-level retrieval** answers "which specific articles say something about this topic?" It embeds each article node (from `document_nodes` where `node_type = 'article'`) as an independent unit and retrieves the most semantically relevant articles.

The LLM then receives the top-K retrieved articles as grounding context and generates an answer. This approach is well-suited to legal work because lawyers think in terms of specific articles, not vague document-level topics.

---

## 4. Embedding Strategy

### 4.1 Model Selection

We recommend **Amazon Titan Embed Text v2** (`amazon.titan-embed-text-v2:0`) as the primary embedding model for the following reasons. It is available natively through AWS Bedrock, which satisfies the AWS-native constraint. It supports Arabic text and was trained on multilingual data including Arabic-script languages. It produces 1,024-dimensional vectors, which is a good balance between precision and storage cost. Finally, unlike third-party models (Cohere, OpenAI), it does not require data to leave the AWS environment, which is critical for sovereign legal documents.

An alternative worth evaluating is **Cohere Embed Multilingual v3** also available through Bedrock. It uses explicit Arabic tokenisation and was trained on a larger Arabic corpus. For a production system, we recommend A/B testing both on a held-out query set drawn from actual legal analyst questions.

### 4.2 What to Embed

The granularity of embedding units should match the granularity of expected queries:

For **article-level** units (the primary retrieval target), embed the article heading + full article text concatenated. Articles in UAE law are typically 50–300 words, which is ideal for embedding — short enough to be semantically focused, long enough to provide context.

For **chapter-level** units (for topic-level queries), embed the chapter heading + the first paragraph of the chapter + a list of article headings within it. This creates a "table of contents" style embedding that enables coarse retrieval before article-level refinement.

For **document-level** units (for cross-law queries), embed the preamble + metadata summary (law number, effective date, subject area, issuing authority).

### 4.3 Embedding Pipeline

The embedding pipeline runs as a post-processing step after Stage 4 of the ingestion pipeline. For each article node in `document_nodes`, we:

1. Retrieve the article text and its parent chapter heading from the database.
2. Construct the embedding input: `{chapter_heading} | {article_heading}\n{article_text}`. Prepending the chapter heading provides crucial context — an article about "exceptions" means very different things in a financial regulation chapter versus a criminal procedure chapter.
3. Call Bedrock's Titan Embed API.
4. Store the resulting vector in Amazon OpenSearch Serverless with k-NN enabled.

Each vector in OpenSearch carries a payload of metadata fields (document_id, node_id, law_number, document_type, jurisdiction, effective_date) enabling pre-filter retrieval — for example, "find relevant articles, but only from in-force laws in Dubai."

---

## 5. Vector Store: Amazon OpenSearch Serverless

We recommend **Amazon OpenSearch Serverless** with k-NN enabled over alternatives like pgvector for the following reasons. OpenSearch is purpose-built for approximate nearest-neighbour search and scales independently of the relational database. It supports pre-filtering on metadata (jurisdiction, document_type, date_range) before the vector search, which is essential in a legal context where most queries have a scope constraint. AWS natively integrates it with IAM for access control and VPC for network isolation. The managed (serverless) option eliminates index management overhead.

The index mapping for articles would look like this:

```json
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": { "name": "hnsw", "space_type": "cosinesimil" }
      },
      "text": { "type": "text", "analyzer": "arabic" },
      "node_id": { "type": "keyword" },
      "document_id": { "type": "keyword" },
      "law_number": { "type": "keyword" },
      "jurisdiction": { "type": "keyword" },
      "status": { "type": "keyword" },
      "effective_date": { "type": "date" },
      "depth": { "type": "integer" }
    }
  }
}
```

The `"analyzer": "arabic"` in the `text` field enables hybrid search: we can combine BM25 keyword relevance (useful for exact law number references) with vector similarity (useful for semantic queries). This hybrid approach reliably outperforms either method alone on legal text.

---

## 6. RAG Query Flow

When a legal analyst submits a query, the system follows this flow:

**Step 1 — Query normalisation.** The analyst's question (which may mix Arabic and English) is normalised using the same Arabic utilities as the ingestion pipeline: alef normalisation, harakat removal, digit conversion. This ensures the query embedding is computed on the same normalised surface form as the indexed articles.

**Step 2 — Query embedding.** The normalised query is embedded using the same Bedrock Titan model used for document embeddings. Critically, we use the same model for both query and document — a common mistake is using different models which produces incompatible vector spaces.

**Step 3 — Filtered k-NN retrieval.** The query vector is submitted to OpenSearch with a pre-filter that constrains results to the analyst's scope. For example, a query about financial regulation might pre-filter `jurisdiction=UAE` and `status=in_force` and `document_type IN (Law, Regulation)`. We retrieve the top-20 candidates.

**Step 4 — Reranking (optional but recommended).** A lightweight reranker — either a cross-encoder or Cohere's Rerank API via Bedrock — scores the top-20 retrieved articles against the original query and returns the top-5. Reranking is especially valuable for Arabic because it operates on the full (query, passage) pair rather than independent embeddings, capturing fine-grained relevance signals.

**Step 5 — LLM synthesis.** The top-K articles, prefaced with their law number, article number, and effective date, are assembled into a prompt and sent to Bedrock Claude. The system prompt instructs the model to answer only from the provided context and to cite the specific article number for each claim. This citation requirement is non-negotiable for legal use — analysts must be able to verify every statement.

**Step 6 — Response with citations.** The response is returned to the analyst with inline citations like [Federal Law No. 3/2022, Article 14] that link back to the original `document_nodes` record.

---

## 7. Key Challenges and Mitigations

**Challenge: cross-reference resolution.** UAE laws frequently reference other laws: "as provided for in Federal Law No. X of Year Y, Article Z." These references are currently stored as raw text strings. The next engineering step would be to run a Named Entity Recognition (NER) pass over all article text to extract and resolve these cross-references into proper foreign-key relationships in the database. This would enable graph-style traversal: "show me all laws that reference or are referenced by Law 3/2022."

**Challenge: legal amendment tracking.** When a law is amended, the superseded articles must not be returned as current answers. The `status` and `effective_date` metadata fields from Stage 4 handle this at the document level, but article-level amendment tracking (where individual articles within a still-in-force law are superseded) requires a separate amendment parsing stage. This is an identified gap in the current v2 pipeline and a recommended extension.

**Challenge: hallucination in legal contexts.** LLMs can generate plausible-sounding but incorrect legal citations. The mitigation is strict prompt engineering (respond only from provided context; never infer law numbers from memory) combined with a post-generation verification step that checks that every cited article number exists in the retrieved context.

**Challenge: dialect and register variation.** UAE legal documents are written in Modern Standard Arabic (فصحى), but analyst queries may include Gulf Arabic colloquialisms. The embedding model partially handles this via semantic similarity, but explicit query expansion — using the LLM to rewrite a query in formal MSA before embedding — can improve recall significantly.

---

## 8. Implementation Roadmap

The recommended phased approach is as follows. Phase 1 (weeks 1–2) implements the embedding pipeline and OpenSearch index, ingesting all existing documents from the warehouse. Phase 2 (weeks 3–4) builds the RAG query API as an AWS Lambda function behind API Gateway, supporting the filtered k-NN retrieval + LLM synthesis flow. Phase 3 (months 2–3) adds the cross-reference NER pass, the hybrid BM25 + vector search, and the reranking step. Phase 4 (ongoing) evaluates answer quality against a legal analyst–curated golden dataset and iterates on the prompt engineering and model selection.

---

## 9. AWS Services Summary

The complete LLM-augmented stack uses the following AWS services in addition to those already used by the ingestion pipeline: Amazon Bedrock (Titan Embed v2 for embeddings, Claude 3.5 Sonnet for synthesis and profiling), Amazon OpenSearch Serverless (k-NN vector index), AWS Lambda (query handler), Amazon API Gateway (REST interface), and Amazon SageMaker (optional: fine-tuning a domain-specific Arabic embedding model in later phases).
