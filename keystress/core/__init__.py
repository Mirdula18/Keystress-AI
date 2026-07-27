"""
Core domain logic: the privacy boundary, feature extraction, model loading, inference.

Nothing in this package imports Flask. The domain is testable without a web server, which
is what makes the privacy guarantee verifiable in isolation.
"""
