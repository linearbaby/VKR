from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


class Base(DeclarativeBase):
    pass


class UserMEM(Base):
    __tablename__ = "user_embeddings"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id_fk: Mapped[int] = mapped_column(ForeignKey("user_account.id"))
    embedding: Mapped[str] = mapped_column(String(12000))


class UserWEB(Base):
    __tablename__ = "user_account"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    fullname: Mapped[Optional[str]] = mapped_column(String(30))


class MusicInfo(Base):
    __tablename__ = "music_info"

    id: Mapped[int] = mapped_column(primary_key=True)
    track_name: Mapped[str] = mapped_column(String(256))
    path: Mapped[str] = mapped_column(String(256))


class MusicEmbedding(Base):
    __tablename__ = "music_embedding"

    id: Mapped[int] = mapped_column(primary_key=True)
    fk_id_music_info: Mapped[int] = mapped_column(ForeignKey("music_info.id"))
    embedding: Mapped[str] = mapped_column(String(12000))
