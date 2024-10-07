package com.imashnake.aigo.ui.features

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import kotlinx.serialization.Serializable

@Composable
fun CameraOverlay(modifier: Modifier = Modifier) {
    Text(
        text = "Camera Permissions granted! Show the camera here right now.",
        color = MaterialTheme.colorScheme.onBackground,
    )
}

@Serializable
data object CameraOverlay
