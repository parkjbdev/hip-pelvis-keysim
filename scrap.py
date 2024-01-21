from bs4 import BeautifulSoup
import requests as req
import json
from ast import literal_eval
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM

# Progress Bars
from tqdm import tqdm
from yaspin import yaspin


def get_html(url):
    res = req.get(url)
    html = res.text
    soup = BeautifulSoup(html, "html.parser")
    return soup


def get_article_urls():
    urls = []

    # Use Cached URL if exists
    if os.path.exists("urls.txt"):
        with open("urls.txt", "r") as f:
            urls = f.read().split("\n")
    else:
        urls = fetch_article_urls()
        with open("urls.txt", "w") as f:
            for url in urls:
                f.write(url + "\n")

    return urls


def fetch_article_urls():
    doc = get_html("https://hipandpelvis.or.kr/index.php?body=archive")
    anchors = doc.select(".accordion-collapse a")

    urls = []

    pbar = tqdm(anchors)

    for anchor in pbar:
        url = f'https://hipandpelvis.or.kr/{anchor["href"]}'
        pbar.set_description(f'fetching {anchor["href"]}')
        doc = get_html(url)
        anchors = doc.select(".ToC_title a")
        for anchor in anchors:
            article_url = f'https://hipandpelvis.or.kr/{anchor["href"]}'
            urls.append(article_url)

    return urls


def fetch_abstract(url):
    doc = get_html(url)
    title = doc.select("h1.content-title")[0].text.strip()
    doiurl = doc.select(".article-meta-doi-link")[0].text.strip()

    # Parse keywords
    keywords = []
    keyword_spans = doc.select("div.article-keyword-group-title + div > span")
    for keyword_span in keyword_spans:
        keywords.append(keyword_span.text.strip())

    # Parse abstract
    abstract = ""
    divs = doc.select(".article-abstract h3, .article-abstract p")
    for div in divs:
        abstract += f"{div.get_text()}\n"

    return {"title": title, "url": doiurl, "abstract": abstract, "keywords": keywords}


class OllamaChatBot:
    base_url = "http://localhost:11434"
    model = "mixtral:latest"

    def __init__(self, base_url="http://localhost:11434", model="mistral:latest"):
        self.base_url = base_url
        self.model = model
        if not self.model_exists(model):
            self.pull(model)

    def model_exists(self, model_name):
        models = list(
            map(
                lambda x: x["name"],
                req.get(f"{self.base_url}/api/tags").json()["models"],
            )
        )
        return model_name in models

    def pull(self, model_name):
        print(f"pulling model {model_name}")

        res_stream = req.post(
            f"{self.base_url}/api/pull",
            json={
                "name": model_name,
            },
            stream=True,
        )

        status = None
        last_status = None
        pbar = None

        for line in res_stream.iter_lines():
            if line:
                result = json.loads(line)

                last_status = status
                status = result["status"]
                if last_status != status:
                    print(status)

                if "total" in result:
                    if pbar is None:
                        pbar = tqdm(total=result["total"])
                    pbar.n = result["completed"]
                    pbar.refresh()

        if pbar is not None:
            pbar.close()

    @yaspin(text="Waiting for ollama server to respond")
    def chat(self, prompt):
        res = req.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
        )
        ans = res.json()
        return ans["response"]


class HuggingFaceChatBot:
    # model_id = "emilyalsentzer/Bio_ClinicalBERT"
    # model_id = "microsoft/biogpt"
    model_id = "stanford-crfm/BioMedLM"

    @yaspin(text="Initializing huggingface model..")
    def __init__(self, model="stanford-crfm/BioMedLM"):
        self.model_id = model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        # model = BioGptForCausalLM.from_pretrained(model_id)
        # tokenizer = BioGptTokenizer.from_pretrained(model_id)
        print(f"Huggingface Model {model} initialized")

    # TODO: integrate huggingface interface
    @yaspin(text="Waiting for huggingface model to respond..")
    def chat_huggingface(self, prompt):
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        text = generator(
            prompt,
            max_length=200,
            # num_return_sequences=5,
            # do_sample=True
        )

        return text[0]["generated_text"]


def compare_keywords(inferred, answer):
    inferred = [keyword.lower() for keyword in inferred]
    answer = [keyword.lower() for keyword in answer]

    inferred = set(inferred)
    answer = set(answer)

    print("compare")
    print(f"inferred: {inferred}")
    print(f"answer: {answer}")
    print(f"intersection: {inferred.intersection(answer)}")


if __name__ == "__main__":
    urls = get_article_urls()
    articles = []

    # Ollama

    # ollama = OllamaChatBot()
    # prompt = "Return the keyword of the following abstract. "
    # prompt_form = "The answer MUST BE ONLY a python array (such as ['keyword1', 'keyword2', 'keyword3']) since the results will be directly delivered to python code. "
    #
    # for url in urls:
    #     article = fetch_abstract(url)
    #
    #     while True:
    #         try:
    #             article["inferred-keywords"] = literal_eval(
    #                 ollama.chat(prompt + prompt_form + article["abstract"])
    #             )
    #         except Exception as e:
    #             print("Exception Occurred. Retrying..")
    #             continue
    #         break
    #
    #     compare_keywords(article["inferred-keywords"], article["keywords"])
    #
    #     articles.append(article)
    #
    #     with open("articles.json", "w") as f:
    #         json.dump(articles, f)
    #

    # HuggingFace

    hugger = HuggingFaceChatBot()

    for url in urls:
        article = fetch_abstract(url)

        while True:
            try:
                hugger_prompt = (
                    article["abstract"]
                    + "\n\nThe main keywords delimited by comma are "
                )
                keywords = hugger.chat_huggingface(hugger_prompt)[len(hugger_prompt):]
                article["inferred-keywords"] = keywords
                print(keywords)
            except Exception as e:
                print("Exception Occurred. Retrying..")
                continue
            break

        compare_keywords(article["inferred-keywords"], article["keywords"])

        articles.append(article)

        with open("articles.json", "w") as f:
            json.dump(articles, f)
