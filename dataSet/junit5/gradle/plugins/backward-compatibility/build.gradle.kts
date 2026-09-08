import junitbuild.extensions.markerCoordinates

plugins {
	`kotlin-dsl`
}

dependencies {
	implementation("junitbuild.base:dsl-extensions")
	implementation(libs.plugins.download.markerCoordinates)
	implementation(libs.jackson.dataformat.yaml)
}
