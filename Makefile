gunicorn:
	gunicorn -w 4 -b 127.0.0.1:5000 --reload wsgi:app

black:
	black -l 120 ariadne/
	black -l 120 scripts/
	black -l 120 tests/
	black -l 120 wsgi.py
