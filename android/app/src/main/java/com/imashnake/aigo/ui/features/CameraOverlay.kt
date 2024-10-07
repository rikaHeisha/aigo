package com.imashnake.aigo.ui.features

import android.content.Context
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.vectorResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.imashnake.aigo.R
import kotlinx.serialization.Serializable
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

@Composable
fun CameraOverlay(modifier: Modifier = Modifier) {
    Box(modifier.fillMaxSize()) {
        // TODO: Copied this code from https://medium.com/@deepugeorge2007travel/mastering-camerax-in-jetpack-compose-a-comprehensive-guide-for-android-developers-92ec3591a189
        //  Properly set this up using: https://developer.android.com/media/camera/camerax/preview.
        val lensFacing = CameraSelector.LENS_FACING_BACK
        val lifecycleOwner = LocalLifecycleOwner.current
        val context = LocalContext.current
        val preview = Preview.Builder().build()
        val previewView = remember { PreviewView(context) }
        val cameraxSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()
        LaunchedEffect(lensFacing) {
            val cameraProvider = context.getCameraProvider()
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(lifecycleOwner, cameraxSelector, preview)
            preview.surfaceProvider = previewView.surfaceProvider
        }

        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        Image(
            imageVector = ImageVector.vectorResource(R.drawable.blank_go_board),
            contentDescription = null,
            modifier = Modifier
                .align(Alignment.Center)
                .padding(24.dp)
                .graphicsLayer { alpha = 0.3f }
        )
    }
}

private suspend fun Context.getCameraProvider(): ProcessCameraProvider =
    suspendCoroutine { continuation ->
        ProcessCameraProvider.getInstance(this).also { cameraProvider ->
            cameraProvider.addListener({
                continuation.resume(cameraProvider.get())
            }, ContextCompat.getMainExecutor(this))
        }
    }

@Serializable
data object CameraOverlay
