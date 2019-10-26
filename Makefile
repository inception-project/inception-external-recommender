gunicorn:
	gunicorn -w 4 -b 127.0.0.1:5000 --reload wsgi:server._app

black:
	black -l 120 ariadne/
	black -l 120 scripts/