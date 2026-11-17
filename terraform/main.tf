provider "datadog" {
  api_key = var.datadog_api_key
  app_key = var.datadog_app_key
}

module "gcp_integration" {
    source = "./google_cloud_integration_terraform"
    gcp_project_id  = var.gcp_project_id
    datadog_api_key = var.datadog_api_key
    datadog_app_key = var.datadog_app_key
}
