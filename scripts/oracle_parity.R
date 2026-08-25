# Gate 2: fit StMoMo Poisson Lee-Carter on the shared SWE panel and dump
# alpha/beta/kappa for comparison against the Python implementation.
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages(library(StMoMo))

root <- dirname(dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))))
pdir <- file.path(root, "results", "parity")
D <- as.matrix(read.csv(file.path(pdir, "D_swe_male.csv"), row.names = 1, check.names = FALSE))
E <- as.matrix(read.csv(file.path(pdir, "E_swe_male.csv"), row.names = 1, check.names = FALSE))
ages <- as.integer(rownames(D)); years <- as.integer(colnames(D))

model <- lc(link = "log")                       # Poisson Lee-Carter (Brouhns)
fit <- fit(model, Dxt = D, Ext = E, ages = ages, years = years, verbose = FALSE, tolerance = 1e-12)

# StMoMo lc() constraints: sum(bx) = 1, sum(kt) = 0 — same convention as
# mortcal.models.lc.PoissonLeeCarter.
out <- data.frame(
  param = c(rep("alpha", length(ages)), rep("beta", length(ages)), rep("kappa", length(years))),
  index = c(ages, ages, years),
  value = c(as.numeric(fit$ax), as.numeric(fit$bx), as.numeric(fit$kt))
)
write.csv(out, file.path(pdir, "stmomo_plc_params.csv"), row.names = FALSE)
cat("StMoMo PLC fitted: loglik", fit$loglik, " deviance", fit$deviance, "\n")
