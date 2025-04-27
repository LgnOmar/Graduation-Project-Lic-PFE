// Import the core Flutter Material library - provides widgets like Scaffold, Text, etc.
import 'package:flutter/material.dart';

// Define LoginScreen as a StatelessWidget (it won't change its own state initially)
class LoginScreen extends StatelessWidget {
  // 'const' constructor for performance optimization if the widget is truly constant
  const LoginScreen({super.key}); // 'key' helps Flutter identify widgets

  // The 'build' method describes how the widget should look
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('JibJob Login'),
        backgroundColor: Colors.blueAccent,
      ),
      body: Padding( // Add padding around the entire form
        padding: const EdgeInsets.all(16.0), // 16 pixels of space on all sides
        child: Column( // Arrange widgets vertically
          mainAxisAlignment: MainAxisAlignment.center, // Center vertically in the Column
          crossAxisAlignment: CrossAxisAlignment.stretch, // Stretch children horizontally
          children: <Widget>[ // 'children' is a list of widgets inside the Column

            // --- Logo (Optional) ---
            // Replace with your actual logo widget later
            Icon(Icons.work, size: 80.0, color: Colors.blueAccent),
            const SizedBox(height: 40.0), // Adds vertical space

            // --- Email Input Field ---
            TextField(
              decoration: InputDecoration(
                labelText: 'Email Address', // Placeholder label
                border: OutlineInputBorder(), // Adds a border
                prefixIcon: Icon(Icons.email), // Icon inside the field
              ),
              keyboardType: TextInputType.emailAddress, // Suggests email keyboard
            ),
            const SizedBox(height: 16.0), // Space between fields

            // --- Password Input Field ---
            TextField(
              decoration: InputDecoration(
                labelText: 'Password',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.lock),
                // Add suffix icon to show/hide password later if needed
              ),
              obscureText: true, // Hides the typed text (dots)
            ),
            const SizedBox(height: 24.0), // More space before the button

            // --- Login Button ---
            ElevatedButton(
              onPressed: () {
                // TODO: Implement login logic later
                print('Login button pressed!'); // Placeholder action
              },
              style: ElevatedButton.styleFrom( // Basic button styling
                backgroundColor: Colors.blueAccent, // Button color
                padding: EdgeInsets.symmetric(vertical: 16.0), // Make button taller
                textStyle: TextStyle(fontSize: 18.0, color: Colors.white), // Text style
              ),
              child: const Text('Login', style: TextStyle(color: Colors.white)), // Button text
            ),
            const SizedBox(height: 16.0),

            // --- Sign Up Link (Optional) ---
            TextButton(
              onPressed: () {
                // TODO: Implement navigation to sign up screen later
                print('Navigate to Sign Up');
              },
              child: Text('Don\'t have an account? Sign Up'),
            ),
          ],
        ),
      ),
    );
  }

}
