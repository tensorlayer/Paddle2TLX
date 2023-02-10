import setuptools
import paddle2tlx

long_description = "paddle2tlx is a toolkit for converting TensorLayerX model project from PaddlePaddle frameworks.\n\n"
long_description += "Usage: paddle2tlx --input_dir_pd pd_project_dir --output_dir_tlx tlx_project_dir\n"
long_description += "GitHub: \n"
long_description += "Email: "

# with open("requirements.txt") as fin:
#     REQUIRED_PACKAGES = fin.read()

setuptools.setup(
    name="paddle2tlx",
    version=paddle2tlx.__version__,
    author="sthg",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="",
    packages=setuptools.find_packages(),
    # install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    entry_points={'console_scripts': ['paddle2tlx=paddle2tlx.convert:main', ]}
)
