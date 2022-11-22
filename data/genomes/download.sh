#!/usr/bin/env bash

for GENOME in "hg19" "hg38" "mm10"; do
    if ! [ -f ${GENOME}/${GENOME}.fa.sizes ]; then
        genomepy install -p UCSC -g ./ -t 8 -f ${GENOME}
    fi
done
