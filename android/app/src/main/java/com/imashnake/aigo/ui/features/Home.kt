package com.imashnake.aigo.ui.features

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.imashnake.aigo.R
import com.imashnake.aigo.ui.components.AigoIconTextButton
import kotlinx.serialization.Serializable

@Composable
fun Home(
    modifier: Modifier = Modifier,
    takePicture: () -> Unit,
) {
    Column(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        AigoIconTextButton(
            drawable = R.drawable.photo_camera,
            text = "Take Picture",
            onClick = takePicture
        )
        AigoIconTextButton(R.drawable.round_image, "Pick Image")
    }
}

@Serializable
data object Home
