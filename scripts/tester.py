import argparse
import json
import urllib.request
from http.client import HTTPResponse
from typing import Any

from cassis import load_typesystem, load_cas_from_xmi


def send_train_request(path_to_json: str, user: str):
    with open(path_to_json) as f:
        json_data = json.load(f)

    for document in json_data["documents"]:
        document["userId"] = user

    response = _send_json("http://localhost:5000/sklearn_sentence/train", json_data)
    print(response.status)
    print(response.reason)


def send_predict_request(path_to_json: str, user: str):
    with open(path_to_json) as f:
        json_data = json.load(f)

    json_data["document"]["userId"] = user
    response = _send_json("http://localhost:5000/sklearn_sentence/predict", json_data)
    body = json.load(response)

    typesystem = load_typesystem(json_data["typeSystem"])
    cas = load_cas_from_xmi(body["document"], typesystem)
    layer = json_data["metadata"]["layer"]
    feature = json_data["metadata"]["feature"]

    for prediction in cas.select(layer):
        if prediction.inception_internal_predicted:
            print(f"{getattr(prediction, feature)}\t\t {cas.get_covered_text(prediction)}")


def _send_json(url: str, body: Any) -> HTTPResponse:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"content-type": "application/json"}
    )
    return urllib.request.urlopen(req)


def main():
    parser = argparse.ArgumentParser(description="Test your INCEpTION external recommender.")
    parser.add_argument("request_type", choices=["train", "predict"], help="The request type you want to use.")
    parser.add_argument("-u", "--user", default="admin", help="The user issuing the request.")
    args = parser.parse_args()

    if args.request_type == "train":
        path_to_json = "examples/requests/training_sentence_sentiment.json"
        send_train_request(path_to_json, args.user)
    elif args.request_type == "predict":
        path_to_json = "examples/requests/predict_sentence_sentiment.json"
        send_predict_request(path_to_json, args.user)
    else:
        raise ValueError(f"Invalid request type: [{args.request_type}]")


if __name__ == "__main__":
    main()
