terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.39"
    }

    datadog = {
      source  = "datadog/datadog"
      version = "3.42.0"
    }
  }
}
