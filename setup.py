from setuptools import setup, find_packages

contrib = [
    'Markus Rempfler',
    'Jan Eglinger',
]

# setup.
setup(
    name='faim-luigi',
    version='0.1.1',
    description='faim-luigi',
    author=', '.join(contrib),
    packages=find_packages(exclude=[
        'tests',
        'examples'
    ]),
    install_requires=[
        'luigi',
        'knime',
        'tifffile>=2020.5.30',
#        'dlutils',  # TODO add version
        'sqlalchemy',
        'pandas',
#        'tensorflow',  # TODO could potentially be replaced by dlutils
        'scikit-image>=0.16.2'
    ])
