import requests

BASE = 'http://127.0.0.1:5000'

s = requests.Session()
print('GET /login ->', s.get(BASE + '/login').status_code)
# try with spaces
r = s.post(BASE + '/login', data={'username':' admin ','password':'password'}, allow_redirects=False)
print('POST with spaces ->', r.status_code, 'Location:', r.headers.get('Location'))
# try different case
r2 = s.post(BASE + '/login', data={'username':'Admin','password':'password'}, allow_redirects=False)
print('POST Admin ->', r2.status_code, 'Location:', r2.headers.get('Location'))
# try wrong password
r3 = s.post(BASE + '/login', data={'username':'admin','password':'wrong'}, allow_redirects=False)
print('POST wrong pw ->', r3.status_code)
