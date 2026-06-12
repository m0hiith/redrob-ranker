# Redrob Candidate Ranker — reproduction container (Stage-3 compatible)
#
# Build (network allowed here — this is the documented pre-computation step;
# it installs pinned deps and vendors the embedding model into the image):
#   docker build -t redrob-ranker .
#
# Rank (no network, CPU only, as the spec requires):
#   docker run --rm --network=none \
#     -v /path/to/candidates.json:/data/candidates.json:ro \
#     -v "$PWD/out:/out" \
#     redrob-ranker \
#     python rank.py --candidates /data/candidates.json --out /out/submission.csv

FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py rank.py self_check.py ./
# Vendor the model at build time so ranking runs offline.
RUN python download_model.py

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CMD ["python", "rank.py", "--help"]
