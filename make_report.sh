#!/bin/bash

pandoc report.md \
	-V classoption=twocolumn \
	-V documentclass=article \
	-o report.pdf
