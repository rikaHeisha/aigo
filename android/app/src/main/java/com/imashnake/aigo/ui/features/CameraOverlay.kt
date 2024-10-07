package com.imashnake.aigo.ui.features

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.serialization.Serializable

@Composable
fun CameraOverlay(modifier: Modifier = Modifier) {
    Box(modifier.fillMaxSize()) {
        Text(
            text = "Camera Permissions granted! Show the camera here right now.",
            color = MaterialTheme.colorScheme.onBackground,
            modifier = Modifier.align(Alignment.Center).padding(16.dp),
        )
    }
}

@Serializable
data object CameraOverlay
