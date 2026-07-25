# Reproduction artifacts

Generated gradient factors, curvature files, score matrices, datasets, and model
checkpoints do not belong in Git. Put them in this directory (or pass another
path to the CLI); everything here is ignored.

LDS does **not** require the 500 subset-trained GPT-2 checkpoints once their
averaged query losses have been computed. See the LDS evaluation section in the
main README.
