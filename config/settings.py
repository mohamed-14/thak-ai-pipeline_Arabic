"""
config/settings.py
──────────────────
Centralised configuration management using Pydantic Settings.
All values can be overridden via environment variables or a .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings.  Every attribute maps 1-to-1 to an environment
    variable of the same name (case-insensitive).  Sensitive values such as
    DB_PASSWORD should be provided via .env or AWS Secrets Manager — never
    hard-coded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── AWS ───────────────────────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: str | None = Field(default=None)
    aws_secret_access_key: str | None = Field(default=None)
    # When running on EC2/ECS with an IAM role the two keys above can be None
    # and boto3 will pick up credentials automatically.

    # ── S3 ────────────────────────────────────────────────────────────────────
    s3_bucket_name: str = Field(
        default="thakai-documents",
        description="S3 bucket that contains the raw legal PDFs",
    )
    s3_laws_prefix: str = Field(default="Laws/arabic/")
    s3_regulatory_prefix: str = Field(default="Regulatory/arabic/")

    # ── Database (Amazon RDS PostgreSQL) ──────────────────────────────────────
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="thakai")
    db_user: str = Field(default="thakai_user")
    db_password: str = Field(default="changeme")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # ── AWS Bedrock ───────────────────────────────────────────────────────────
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        description="Bedrock model used for document profiling and metadata extraction",
    )
    bedrock_embed_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="Bedrock model used to generate text embeddings",
    )
    bedrock_max_tokens: int = Field(default=4096)

    # ── Amazon Textract ───────────────────────────────────────────────────────
    textract_confidence_threshold: float = Field(
        default=70.0,
        description="Minimum Textract confidence score (%) to accept an OCR block",
    )

    # ── Pipeline behaviour ────────────────────────────────────────────────────
    # The minimum number of meaningful Arabic characters on a page needed for
    # PyMuPDF extraction to be considered successful (below this we OCR).
    min_arabic_chars_for_direct_extract: int = Field(default=100)

    # How many pages to send to the LLM for document profiling.
    profiling_sample_pages: int = Field(default=5)

    log_level: str = Field(default="INFO")


# Module-level singleton — import this everywhere.
settings = Settings()
