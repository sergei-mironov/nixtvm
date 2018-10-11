import certifi
import requests

try:
    print('Checking connection to AWS...')
    test = requests.get('https://s3.amazonaws.com/img-datasets/mnist.npz')
    print('Connection to Slack OK.')

except requests.exceptions.SSLError as err:
    print('SSL Error. Adding custom certs to Certifi store...')
    cafile = certifi.where()
    with open('/etc/ssl/certs/ca-bundle.crt', 'rb') as infile:
        customca = infile.read()
    with open(cafile, 'ab') as outfile:
        outfile.write(customca)
    print('That might have worked.')
