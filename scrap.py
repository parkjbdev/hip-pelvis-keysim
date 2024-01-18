from bs4 import BeautifulSoup
import requests as req
import json
from ast import literal_eval
import os


def get_html(url):
    res = req.get(url)
    html = res.text
    soup = BeautifulSoup(html, "html.parser")
    return soup


def get_article_urls():
    doc = get_html("https://hipandpelvis.or.kr/index.php?body=archive")
    anchors = doc.select(".accordion-collapse a")

    urls = []

    for anchor in anchors:
        url = f'https://hipandpelvis.or.kr/{anchor["href"]}'
        doc = get_html(url)
        anchors = doc.select(".ToC_title a")
        for anchor in anchors:
            article_url = f'https://hipandpelvis.or.kr/{anchor["href"]}'
            print(article_url)
            urls.append(article_url)

    return urls


def get_abstract(url):
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


def chatbot(prompt):
    res = req.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            # 'model': 'llama2-uncensored',
            "prompt": prompt,
            "stream": False,
        },
    ).text
    ans = json.loads(res)["response"]
    return ans


if __name__ == "__main__":
    urls = []

    if os.path.exists("urls.txt"):
        with open("urls.txt", "r") as f:
            urls = f.read().split("\n")
    else:
        urls = get_article_urls()
        with open("urls.txt", "w") as f:
            for url in urls:
                f.write(url + "\n")

    articles = []

    prompt = "Return the keyword of the following abstract. "
    prompt_form = "The answer MUST BE ONLY a python array (such as ['keyword1', 'keyword2', 'keyword3']) since the results will be directly delivered to python code. "

    for url in urls:
        article = get_abstract(url)

        while True:
            try:
                article["inferred-keywords"] = literal_eval(
                    chatbot(prompt + prompt_form + article["abstract"])
                )
            except Exception as e:
                print(e)
                continue
            break

        print(article)
        articles.append(article)

        with open("articles.json", "w") as f:
            json.dump(articles, f)
