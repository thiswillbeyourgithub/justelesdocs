"""What the encoder accepts, shared by the scripts that prepare text for it.

`chunk.py` decides how long a passage may be and `embed.py` decides how long a
passage may be fed, and the two numbers have to be one number: a chunker that
lets a section run past what the encoder reads would ship a vector embedded from
the section's first half only, silently. Both scripts import this rather than
each holding a copy.

Written by Claude Code (Fable 5.1).
"""

# Longest passage, in tokens, fed to the encoder. 8192 is the model's own limit
# (8194 positions, two of them sentinels), so nothing in this corpus is truncated
# at all: the longest page is 4481 tokens and the longest packed chunk 360. The
# section chunker (`chunk.py --section-chunks`) is the one producer that can
# exceed it, and splits a section that does; `embed.py` truncates at it as the
# last line of defence and reports how many rows reached the ceiling.
MAX_PASSAGE_TOKENS = 8192
