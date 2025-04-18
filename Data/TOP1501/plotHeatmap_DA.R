plotHeatmap_DA <- function(out = NULL, analysis_type = c("DA"), top_n = 20, threshold = 0.1, 
                        res = NULL, d_se = NULL, d_counts = NULL, d_medians = NULL, d_medians_by_cluster_marker = NULL, 
                        sample_order = NULL, show_column_names = FALSE, 
                        save_dir = "~/Desktop/AutoEncoderCyTOF/TOP1501/",
                        img_name = "Diffcyt1.png") {
  
  if (!is.null(out)) {
    res <- out$res
    d_se <- out$d_se
    d_counts <- out$d_counts
  }
  if (is.null(out)) {
    d_medians <- calcMedians(d_se)
    d_medians_by_cluster_marker <- calcMediansByClusterMarker(d_se)
  } else  {
    d_medians <- out$d_medians
    d_medians_by_cluster_marker <- out$d_medians_by_cluster_marker
  }
  
  analysis_type <- match.arg(analysis_type, choices = c("DA", "DS"))
  
  
  is_marker <- SummarizedExperiment::colData(d_medians_by_cluster_marker)$marker_class != "none"
  is_celltype_marker <- SummarizedExperiment::colData(d_medians_by_cluster_marker)$marker_class == "type"
  is_state_marker <- SummarizedExperiment::colData(d_medians_by_cluster_marker)$marker_class == "state"
  
  
  # ------------------------------------------------------------------------------
  # heatmap: main panel (DA tests and DS tests): expression of 'cell type' markers
  # ------------------------------------------------------------------------------
  
  d_heatmap <- 
    SummarizedExperiment::assay(d_medians_by_cluster_marker)[, is_marker, drop = FALSE]
  
  d_heatmap_celltype <- 
    SummarizedExperiment::assay(d_medians_by_cluster_marker)[, is_celltype_marker, drop = FALSE]
  
  # arrange alphabetically
  d_heatmap_celltype <- d_heatmap_celltype[, order(colnames(d_heatmap_celltype)), drop = FALSE]
  
  if (analysis_type == "DA") {
    stopifnot(nrow(d_heatmap) == length(SummarizedExperiment::rowData(res)$cluster_id), 
              nrow(d_heatmap_celltype) == length(SummarizedExperiment::rowData(res)$cluster_id), 
              all(rownames(d_heatmap) == SummarizedExperiment::rowData(res)$cluster_id), 
              all(rownames(d_heatmap_celltype) == SummarizedExperiment::rowData(res)$cluster_id))
  }
  
  
  # results for top clusters
  d_top <- topTable(res, top_n = top_n, format_vals = FALSE)
  
  # top 'n' clusters
  d_heatmap_celltype <- d_heatmap_celltype[match(d_top$cluster_id, rownames(d_heatmap_celltype)), , drop = FALSE]
  
  stopifnot(nrow(d_heatmap_celltype) == nrow(d_top), 
            all(rownames(d_heatmap_celltype) == d_top$cluster_id))
  
  # color scale: 1%, 50%, 99% percentiles across all medians and all markers
  colors <- circlize::colorRamp2(
    quantile(SummarizedExperiment::assay(d_medians_by_cluster_marker)[, is_marker], 
             c(0.01, 0.5, 0.99), na.rm = TRUE), 
    c("royalblue3", "white", "tomato2")
  )
  
  # note: no additional scaling (using asinh-transformed values directly)
  ht_main <- ComplexHeatmap::Heatmap(
    d_heatmap_celltype, col = colors, name = "expression", 
    row_title = "clusters", row_title_gp = gpar(fontsize = 14), 
    column_title = "markers (cell type)", column_title_side = "bottom", column_title_gp = gpar(fontsize = 14), 
    column_names_gp = gpar(fontsize = 6), 
    heatmap_legend_param = list(title_gp = gpar(fontface = "bold", fontsize = 12), labels_gp = gpar(fontsize = 12)), 
    cluster_columns = FALSE, row_names_side = "left", row_names_gp = gpar(fontsize = 11), 
    clustering_distance_rows = "euclidean", clustering_method_rows = "median"
  )
  
  
  # --------------------------------------------------------------
  # heatmap: second panel (DA tests): cluster abundances by sample
  # --------------------------------------------------------------
  
  if (analysis_type == "DA") {
    
    stopifnot(nrow(d_counts) == nrow(SummarizedExperiment::rowData(res)), 
              all(rownames(SummarizedExperiment::assay(d_counts)) == rownames(SummarizedExperiment::rowData(res))))
    
    d_abundance <- SummarizedExperiment::assay(d_counts)[d_top$cluster_id, , drop = FALSE]
    
    stopifnot(nrow(d_abundance) == nrow(d_heatmap_celltype), 
              all(rownames(d_abundance) == rownames(d_heatmap_celltype)))
    
    # color scale: full range
    colors_counts <- circlize::colorRamp2(range(d_abundance), c("#132a13", "yellow"))
    
    
    # note: row ordering is automatically matched when multiple heatmaps are combined
    ht_abundance <- ComplexHeatmap::Heatmap(
      d_abundance, col = colors_counts, name = "n_cells", column_split = as.factor(res$group_id),
      column_title = "samples", column_title_side = "bottom", column_title_gp = gpar(fontsize = 14), 
      column_names_gp = gpar(fontsize = 12), 
      heatmap_legend_param = list(title_gp = gpar(fontface = "bold", fontsize = 12), labels_gp = gpar(fontsize = 12)), 
      column_order = sample_order, cluster_columns = FALSE, 
      show_row_names = FALSE, show_column_names = show_column_names
    )
  }
  
  
  # ---------------------------------
  # row annotation: adjusted p-values
  # ---------------------------------
  
  # identify column of adjusted p-values
  ix_p_adj <- which(colnames(d_top) == "p_adj")
  
  # significant differential clusters or cluster-marker combinations
  sig <- d_top[, ix_p_adj] <= threshold
  # set filtered clusters or cluster-marker combinations to FALSE
  sig[is.na(sig)] <- FALSE
  
  # set up data frame
  if (analysis_type == "DA") {
    d_sig <- data.frame(cluster_id = d_top$cluster_id, 
                        sig = as.numeric(sig))
  } 
  
  stopifnot(nrow(d_sig) == nrow(d_top), 
            nrow(d_sig) == nrow(d_heatmap_celltype))
  
  # add row annotation
  row_annot <- data.frame(
    "significant" = factor(d_sig$sig, levels = c(0, 1), labels = c("no", "yes")), 
    check.names = FALSE
  )
  
  ha_row <- ComplexHeatmap::rowAnnotation(
    df = row_annot, 
    col = list("significant" = c("no" = "gray90", "yes" = "red")),
    annotation_legend_param = list(title_gp = gpar(fontface = "bold", fontsize = 12), labels_gp = gpar(fontsize = 12)), 
    show_annotation_name = FALSE 
  )
  
  
  # ----------------------
  # combine heatmap panels
  # ----------------------
  
  # title
  if (analysis_type == "DA") {
    ht_title <- "Results: top DA clusters"
  } 
  # combine elements of heatmap
  
  if (analysis_type == "DA") {
    new_ht <- ComplexHeatmap::add_heatmap(ht_main, ht_abundance, direction = "horizontal")
    new_ht <- ComplexHeatmap::add_heatmap(new_ht, ha_row, direction = "horizontal")
    #png(file = file.path(save_dir, img_name))
    ComplexHeatmap::draw(new_ht, 
         column_title = ht_title, column_title_gp = gpar(fontface = "bold", fontsize = 12), 
         auto_adjust = TRUE)
    #dev.off()
  } 
  
}


