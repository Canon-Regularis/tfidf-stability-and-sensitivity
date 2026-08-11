"""User-profile documents and query construction modes (section 3, 7.1)."""

from tfidf_stability.profiles.query_modes import (
    Query,
    QueryMode,
    QuerySet,
    item_as_query,
    leave_one_out_queries,
    user_profile_queries,
)
from tfidf_stability.profiles.user_profile import (
    Interaction,
    ProfileAggregation,
    UserProfile,
    build_profile,
    eligible_users,
    embed_profile,
    group_interactions,
    profile_norm,
)

__all__ = [
    "Interaction",
    "ProfileAggregation",
    "Query",
    "QueryMode",
    "QuerySet",
    "UserProfile",
    "build_profile",
    "eligible_users",
    "embed_profile",
    "group_interactions",
    "item_as_query",
    "leave_one_out_queries",
    "profile_norm",
    "user_profile_queries",
]
