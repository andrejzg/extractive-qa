mkdir data
curl https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -o data/train-v1.1.json
curl https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -o data/dev-v1.1.json
curl https://worksheets.codalab.org/rest/bundles/0xc83bf36cf8714819ba11802b59cb809e/contents/blob/ -o data/squad-dev-evaluate-in1
curl https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ -o evaluate-v1.1.py
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
pip install requirements.txt
