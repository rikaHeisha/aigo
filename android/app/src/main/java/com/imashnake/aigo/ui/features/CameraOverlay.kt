package com.imashnake.aigo.ui.features

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale
import kotlinx.serialization.Serializable

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraOverlay(modifier: Modifier = Modifier) {
    val cameraPermissionState = rememberPermissionState(android.Manifest.permission.CAMERA)

    Box(modifier.fillMaxSize()) {
        when {
            cameraPermissionState.status.isGranted -> {
                Text(
                    text = "Camera Permissions granted! Show the camera here right now.",
                    color = MaterialTheme.colorScheme.onBackground,
                )
            }
            cameraPermissionState.status.shouldShowRationale -> {
                Text(
                    text = "The app needs to see the board!",
                    color = MaterialTheme.colorScheme.onBackground,
                )
            }
            else -> Text(
                text = "Please grant camera permissions!",
                color = MaterialTheme.colorScheme.onBackground,
            )
        }
    }
}

@Serializable
data object CameraOverlay
