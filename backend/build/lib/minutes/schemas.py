"""Structured output the model must return for each meeting.

Items point back to the transcript by *segment number* rather than a clock time:
the model copies an integer it can see, and the server turns it into an exact
timestamp, so citations never drift.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    task: str = Field(description="What needs to be done, phrased as an imperative.")
    owner: str = Field(description="Person responsible, or 'Unassigned'.")
    due: str = Field(description="Deadline as stated in the meeting, or '' if none was given.")
    segment: int = Field(description="Number of the transcript segment where it was agreed.")


class Decision(BaseModel):
    decision: str
    segment: int


class Topic(BaseModel):
    name: str
    first_segment: int
    last_segment: int


class MeetingAnalysis(BaseModel):
    title: str = Field(description="Short, specific meeting title (max 8 words).")
    summary: str = Field(description="3-5 sentence executive summary.")
    key_points: list[str]
    decisions: list[Decision]
    action_items: list[ActionItem]
    open_questions: list[str]
    topics: list[Topic]
