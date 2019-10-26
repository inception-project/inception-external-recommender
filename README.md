# inception-external-recommender

This repository provides **[ariadne](https://inception.fandom.com/wiki/Ariadne)**, a library 
to implement external recommender support for INCEpTION using Python.

## Starting a simple recommender

The following starts a server with two recommender, one for named entities and one for
parts-of-speech. They both use [spaCy](https://spacy.io/). They are not trainable.

    from ariadne.contrib.sklearn import SklearnSentenceClassifier
    from ariadne.server import Server
      
    server = Server()
    server.add_classifier("spacy_ner", SpacyNerClassifier("en"))
    server.add_classifier("spacy_pos", SpacyPosClassifier("en"))

    server.start()
    
## Building your own recommender

See `ariadne/contrib/sklearn.py` for examples.

## Deployment

In order to support multiple users at once, the recommender server needs to be started on
a wsgi server. This can e.g. be done via [gunicorn](https://gunicorn.org/). We provide an
example in `wsgi.py` which can be run on `gunicorn` via

    gunicorn -w 4 -b 127.0.0.1:5000 wsgi:server._app
    
This runs the recommendation server with 4 workers, that means at least 4 users can use the 
server at the same time. Make sure to 

## Testing

**inception-recommender** comes with example requests and a testing script. 

### Tester

The tester allows to send different requests to your external recommender, thereby you
do not need to run INCEpTION during (early) development.

    $ python scripts/tester.py train -h
    usage: tester.py [-h] [-u USER] {train,predict}
    
    Test your INCEpTION external recommender.
    
    positional arguments:
      {train,predict}       The request type you want to use.
    
    optional arguments:
      -h, --help            show this help message and exit
      -u USER, --user USER  The user issuing the request.