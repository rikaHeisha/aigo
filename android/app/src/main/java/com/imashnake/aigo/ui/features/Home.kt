package com.imashnake.aigo.ui.features

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.imashnake.aigo.R
import com.imashnake.aigo.ui.components.AigoIconTextButton
import kotlinx.serialization.Serializable

@Composable
fun Home(
    modifier: Modifier = Modifier,
    onCameraRequested: () -> Unit,
) {
    Column(modifier.fillMaxSize()) {
        AigoIconTextButton(
            drawable = R.drawable.photo_camera,
            text = "Take Picture",
            onClick = onCameraRequested
        )
        AigoIconTextButton(R.drawable.round_image, "Pick Image")
    }
}

@Serializable
data object Home
