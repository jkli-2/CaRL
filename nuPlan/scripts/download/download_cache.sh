# NOTE: Please check the LICENSE file when downloading the cached data. All licenses apply.
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE

mkdir -p dev_v5
for split in {0..3}; do
    wget https://huggingface.co/datasets/autonomousvision/CaRL/resolve/main/dev_v5_${split}.zip
    echo "Extracting file dev_v5_${split}.zip"
    unzip -q dev_v5_${split}.zip
    rm dev_v5_${split}.zip

    rsync -rv dev_v5_${split}/* dev_v5/
    rm -r dev_v5_${split}
done