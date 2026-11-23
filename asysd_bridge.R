# Genuine SearchBridge bridge to the official ASySD R package.
# Usage: Rscript asysd_bridge.R citations.csv unique.csv manual.csv
# Install in R: remotes::install_github("camaradesuk/ASySD")
# ASySD package version should be reported alongside runs for reproducibility.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: Rscript asysd_bridge.R INPUT.csv UNIQUE.csv MANUAL.csv")
}
if (!requireNamespace("ASySD", quietly = TRUE)) {
  stop("The ASySD R package is not installed. Install camaradesuk/ASySD.")
}
citations <- utils::read.csv(
  args[[1]], stringsAsFactors = FALSE, check.names = FALSE,
  fileEncoding = "UTF-8-BOM", na.strings = character(0)
)
required <- c("record_id", "title", "author", "year", "journal", "doi", "abstract", "source")
missing <- setdiff(required, names(citations))
if (length(missing)) {
  stop("ASySD input lacks columns: ", paste(missing, collapse = ", "))
}
if (anyDuplicated(citations$record_id)) {
  stop("ASySD input record_id values must be unique.")
}
result <- ASySD::dedup_citations(
  citations,
  manual_dedup = TRUE,
  merge_citations = TRUE,
  extra_merge_fields = NULL,
  user_input = 1
)
if (!is.list(result) || is.null(result$unique)) {
  stop("Unexpected ASySD output: missing unique dataframe")
}
unique <- result$unique
if (!"record_ids" %in% names(unique) &&
    !"record_id" %in% names(unique) &&
    !"duplicate_id" %in% names(unique)) {
  stop("ASySD did not emit any audit-ready record identifier column")
}
manual <- result$manual_dedup
if (is.null(manual)) manual <- data.frame()
utils::write.csv(unique, file = args[[2]], row.names = FALSE, na = "")
utils::write.csv(manual, file = args[[3]], row.names = FALSE, na = "")
cat("ASySD version: ", as.character(utils::packageVersion("ASySD")), "\n", sep = "")
cat("ASySD input: ", nrow(citations), "\n", sep = "")
cat("ASySD unique groups: ", nrow(unique), "\n", sep = "")
cat("ASySD manual candidate rows: ", nrow(manual), "\n", sep = "")
