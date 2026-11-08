# Installs the R packages used by titanic_model.R.
# The rocker/r-ver base image pins CRAN to a dated snapshot, so install.packages()
# always installs the same package versions -> reproducible builds.

packages <- c("dplyr")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# Fail the Docker build loudly if anything did not install.
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop("Failed to install: ", paste(missing, collapse = ", "))
}

for (pkg in packages) {
  cat(sprintf("Installed %s %s\n", pkg, as.character(packageVersion(pkg))))
}
