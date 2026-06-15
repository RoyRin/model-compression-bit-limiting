#!/bin/bash
# Download and prepare enwiki9 dataset for compression benchmarking
#
# enwiki9 is the first 10^9 bytes (1GB) of English Wikipedia XML dump
# from March 3, 2006. It's a standard compression benchmark dataset.
#
# Usage:
#   bash scripts/download_enwiki9.sh
#
# Output:
#   data/enwiki9/enwiki-20060303-pages-articles.xml.bz2  (compressed download)
#   data/enwiki9/enwiki9.xml                              (first 1GB extracted)
#   data/enwiki9/enwiki9.txt                              (cleaned text)

set -e  # Exit on error

# Create data directory
DATA_DIR="data/enwiki9"
mkdir -p "$DATA_DIR"

echo "=========================================="
echo "Downloading enwiki9 dataset"
echo "=========================================="
echo ""

# URL for the Wikipedia dump (using mattmahoney.net mirror - standard compression benchmark source)
URL="http://mattmahoney.net/dc/enwik9.zip"
COMPRESSED_FILE="$DATA_DIR/enwik9.zip"

# Download if not already present
if [ -f "$COMPRESSED_FILE" ]; then
    echo "✓ Compressed file already exists: $COMPRESSED_FILE"
else
    echo "Downloading from: $URL"
    echo "This may take several minutes (file is ~25MB compressed)..."
    curl -L -o "$COMPRESSED_FILE" "$URL"
    echo "✓ Download complete"
fi

echo ""
echo "File info:"
ls -lh "$COMPRESSED_FILE"
echo ""

# Extract enwik9 (standard 1GB benchmark file)
ENWIK9_FILE="$DATA_DIR/enwik9"

if [ -f "$ENWIK9_FILE" ]; then
    echo "✓ enwik9 already exists: $ENWIK9_FILE"
else
    echo "Extracting enwik9 (1GB benchmark file)..."
    # Unzip the file
    unzip -q "$COMPRESSED_FILE" -d "$DATA_DIR"
    echo "✓ Extracted to: $ENWIK9_FILE"
fi

echo ""
echo "File info:"
ls -lh "$ENWIK9_FILE"
echo ""

# Quick stats
echo "=========================================="
echo "Dataset Statistics"
echo "=========================================="
echo "Size: $(wc -c < "$ENWIK9_FILE") bytes (should be 1,000,000,000)"
echo "Lines: $(wc -l < "$ENWIK9_FILE")"
echo ""
echo "First 5 lines:"
head -n 5 "$ENWIK9_FILE"
echo ""
echo "✓ enwiki9 dataset ready for compression benchmarking!"
echo ""
echo "Next steps:"
echo "  1. Create compression dataset: python scripts/create_enwiki9_dataset.py"
echo "  2. Run baselines: python scripts/measure_baselines.py data/compression_dataset_enwiki9.yaml"
echo "  3. Run model compression: python scripts/measure_compression.py 8b data/compression_dataset_enwiki9.yaml"
