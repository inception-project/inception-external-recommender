import argparse
import json
import urllib.request


def send_train_request(path_to_json: str, user: str):
    with open(path_to_json) as f:
        json_data = json.load(f)

    for document in json_data["documents"]:
        document["userId"] = user

    url = " http://localhost:5000/sklearn_sentence/train"
    req = urllib.request.Request(
        url, data=json.dumps(json_data).encode("utf-8"), headers={"content-type": "application/json"}
    )
    response = urllib.request.urlopen(req)
    print(response.info())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test your INCEpTION external recommender.")
    parser.add_argument("request_type", choices=["train", "predict"], help="The request type you want to use.")
    parser.add_argument("-u", "--user", default="admin", help="The user issuing the request.")
    args = parser.parse_args()

    if args.request_type == "train":
        path_to_json = "examples/requests/training_sentence_sentiment.json"
        send_train_request(path_to_json, args.user)
    elif args.request_type == "predict":
        path_to_json = "examples/requests/predict_wikigold_ner.json"
    else:
        raise ValueError(f"Invalid request type: [{args.request_type}]")
