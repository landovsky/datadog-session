output "gcp_service_account_email" {
  value = google_service_account.datadog_integration.email
}
