#!/usr/bin/bash

SPANBERT_PATH='./models/spanbert'

if [[ -d "$SPANBERT_PATH" ]];then
    echo "$SPANBERT_PATH already exist"
    echo 'nothing to do. exitting...'
    exit
fi
echo '[warning] the script will download SpanBERT (190Mo) in the models folder.'
echo 'press <ENTER> to continue...'
read
mkdir './models/spanbert'
cd './models/spanbert'
wget 'https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz'
tar -xvf './spanbert_hf_base.tar.gz'
rm -f './spanbert_hf_base.tar.gz'