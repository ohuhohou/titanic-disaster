# R container for the Titanic logistic regression.
# Build from the REPO ROOT:  docker build -t titanic-r -f src/r_app/Dockerfile .
# rocker/r-ver pins both the R version and a dated CRAN snapshot (reproducible installs).
FROM rocker/r-ver:4.4.1

WORKDIR /app

# Install R packages first so this layer is cached between code changes.
COPY src/r_app/install_packages.R ./src/r_app/install_packages.R
RUN Rscript src/r_app/install_packages.R

# Copy the code (the data is NOT baked into the image - it is mounted at run time).
COPY src/r_app/ ./src/r_app/

ENV DATA_DIR=/app/src/data

CMD ["Rscript", "src/r_app/titanic_model.R"]
