mkdir data
curl https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -o data/train-v1.1.json
curl https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -o data/dev-v1.1.json
curl https://worksheets.codalab.org/rest/bundles/0xc83bf36cf8714819ba11802b59cb809e/contents/blob/ -o data/squad-dev-evaluate-in1
curl https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ -o evaluate-v1.1.py
mv evaluate-v1.1 evaluate.py
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
curl -LO http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip -o data/corenlp.zip
cd data
unzip corenlp.zip %% cd stanford-corenlp-full-2018-02-27
pip install -U https://github.com/stanfordnlp/python-stanford-corenlp/archive/master.zip
export CORENLP_HOME=`pwd`
cd ..
# we need python-dev tools for spacy
sudo apt-get install build-essential python3.6-dev
pip install -r requirements.txt
# download resources for spacy and nltk
python -m spacy download en
python -c "import nltk; nltk.download('punkt')"
