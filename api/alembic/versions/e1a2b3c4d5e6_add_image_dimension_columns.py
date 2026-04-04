"""add image dimension columns

Revision ID: e1a2b3c4d5e6
Revises: d5e6f7a8b9c0
Create Date: 2026-04-02 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "d5e6f7a8b9c0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add image dimension and multimodal eligibility columns."""
    op.add_column("files", sa.Column("width", sa.Integer(), nullable=True))
    op.add_column("files", sa.Column("height", sa.Integer(), nullable=True))
    op.add_column("files", sa.Column("original_width", sa.Integer(), nullable=True))
    op.add_column("files", sa.Column("original_height", sa.Integer(), nullable=True))
    op.add_column("files", sa.Column("multimodal_eligible", sa.Boolean(), nullable=True))


def downgrade() -> None:
    """Remove image dimension and multimodal eligibility columns."""
    op.drop_column("files", "multimodal_eligible")
    op.drop_column("files", "original_height")
    op.drop_column("files", "original_width")
    op.drop_column("files", "height")
    op.drop_column("files", "width")
