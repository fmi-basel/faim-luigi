from setuptools import setup, find_packages

contrib = [
    'Markus Rempfler',
]

# setup.
setup(
    name='faim-luigi',
    version='0.0.1',
    description='faim-luigi',
    author=', '.join(contrib),
    packages=find_packages(exclude=[
        'tests',
        'examples'
    ]),
    install_requires=[
        'luigi',
        'keras',
        'dlutils',  # TODO add version
        'json',
        'hashlib',
        'sqlalchemy',
        'tensorflow',  # TODO could potentially be replaced by dlutils
        'scikit-image'
    ])
