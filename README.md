# inception-external-recommender

[![Build Status](https://travis-ci.org/inception-project/inception-external-recommender.svg?branch=master)](https://travis-ci.org/inception-project/inception-external-recommender)
[![codecov](https://codecov.io/gh/inception-project/inception-external-recommender/branch/master/graph/badge.svg)](https://codecov.io/gh/inception-project/inception-external-recommender)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository provides **[ariadne](https://inception.fandom.com/wiki/Ariadne)**, a library 
to run and implement external recommenders for INCEpTION using Python.

## Starting a simple recommender

The following starts a server with two recommender, one for named entities and one for
parts-of-speech. They both use [spaCy](https://spacy.io/). They are not trainable.

    from ariadne.contrib.sklearn import SklearnSentenceClassifier
    from ariadne.server import Server
      
    server = Server()
    server.add_classifier("spacy_ner", SpacyNerClassifier("en"))
    server.add_classifier("spacy_pos", SpacyPosClassifier("en"))

    server.start()
    
They are then reachable under `http://localhost:5000/spacy_ner` and 
`http://localhost:5000/spacy_pos` respectively.
    
## Building your own recommender

See `ariadne/contrib/sklearn.py` for examples.

## Deployment

In order to support multiple users at once, the recommender server needs to be started on
a wsgi server. This can e.g. be done via [gunicorn](https://gunicorn.org/). We provide an
example in `wsgi.py` which can be run on `gunicorn` via

    gunicorn -w 4 -b 127.0.0.1:5000 wsgi:server._app
    
This runs the recommendation server with 4 workers, that means at least 4 users can use the 
server at the same time. Make sure to scale this to your needs. Also adjust the IP adress
the server is listening on. `0.0.0.0` exposes it to your network!

## Contrib Models

Many different models have been already implemented and are ready for you to use:


### Jieba Segmenter

This recommender uses [Jieba](https://github.com/fxsjy/jieba) for predicting Chinese segmentation.

<p align="center">
  <img src="img/jieba.png">
</p>

### S-BERT sentence classifier

This recommender uses [S-BERT](https://github.com/UKPLab/sentence-transformers) together with
[LightGBM](https://lightgbm.readthedocs.io/en/latest/) for sentence classification.

<p align="center">
  <img src="img/sbert_sls.png">
</p>

## Development

The following section describes how to develop your own recommender. **inception-recommender** 
comes with example requests which can be found in `examples/requests`.

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
      
### Developing in deployment setting

The simplest way to develop in deployment setting, that is using `gunicorn` is to just run

    make gunicorn
    
This starts `gunicorn` with 4 workers and hot-code reloading.
