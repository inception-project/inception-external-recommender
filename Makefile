gunicorn:
	gunicorn -w 4 -b 127.0.0.1:5000 wsgi:server._app

black:
	black -l 120 inception-recommender/
	black -l 120 scripts/