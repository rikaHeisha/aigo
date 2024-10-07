package com.imashnake.aigo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.rememberPermissionState
import com.imashnake.aigo.ui.features.CameraOverlay
import com.imashnake.aigo.ui.features.CameraPermissionDialog
import com.imashnake.aigo.ui.features.Home
import com.imashnake.aigo.ui.theme.AigoTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            AigoTheme {
                MainScreen(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(MaterialTheme.colorScheme.background)
                )
            }
        }
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun MainScreen(modifier: Modifier = Modifier) {
    val navController = rememberNavController()
    val isDialogShown = remember { mutableStateOf(false) }
    val cameraPermissionState = rememberPermissionState(
        permission = android.Manifest.permission.CAMERA,
        onPermissionResult = { granted ->
            if (granted) {
                navController.navigate(CameraOverlay)
            } else {
                isDialogShown.value = true
            }
        }
    )

    Box(modifier.fillMaxSize()) {
        if (isDialogShown.value) {
            CameraPermissionDialog(
                permissionState = cameraPermissionState,
                onDismiss = { isDialogShown.value = false },
                launchRequest = cameraPermissionState::launchPermissionRequest,
            )
        }
    }

    NavHost(
        navController = navController,
        startDestination = Home,
        modifier = modifier,
    ) {
        composable<Home> {
            Home(takePicture = cameraPermissionState::launchPermissionRequest)
        }
        composable<CameraOverlay> {
            CameraOverlay()
        }
    }
}
