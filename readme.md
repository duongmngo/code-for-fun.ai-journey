# Setup local environment

## Create virtual environment
python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

## Create .env file
```
OPENAI_API_KEY=
PINECONE_API_KEY=
```