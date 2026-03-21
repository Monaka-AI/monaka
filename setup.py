import setuptools

with open("README.md","r",encoding="utf-8") as r:
    long_description=r.read()

setuptools.setup(
    name="monaka",
    version="0.0.5",
    description="A Japanese parser (including support for historical Japanese)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Monaka-AI/monaka",
    license="MIT",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: BSD 3-Clause License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Japanese",
    ],
    package_data={
        "monaka":["./resource/mecabrc"],
    },
    entry_points={
        "console_scripts":[
            "monaka_train = monaka.train_cli:app",
            "monaka = monaka.cli:app",
            "monaka_server = monaka.server:cmd"
        ],
    },
    project_urls={
        "Source":"https://github.com/Monaka-AI/monaka",
        "Tracker":"https://github.com/Monaka-AI/monaka/issues",
    },
    install_requires=[
        "numpy<2.0.0",
        "protobuf",
        "transformers",
        "registrable",
        "mecab-python3",
        "fugashi",
        "ipadic",
        "typer",
        "torch",
        "prettytable",
        "Flask",
        "requests"
    ],
    extras_require={
        "Train": [
            "tensorboard",
            "tqdm"
        ]
    }
)
