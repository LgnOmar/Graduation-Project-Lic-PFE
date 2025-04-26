import 'package:flutter/material.dart';
// Import your next screen if you want to navigate later
import 'login_screen.dart'; // Example

class SplashScreen extends StatelessWidget {
  const SplashScreen({super.key}); // Standard constructor

  @override
  Widget build(BuildContext context) {
    // Use MediaQuery to get screen height for potentially responsive spacing
    final screenHeight = MediaQuery.of(context).size.height;

    return Scaffold(
      // Set background color matching Figma (adjust Colors.grey[50] if needed)
      // You might need a more specific Color like Color(0xFFF5F5F5) for exact match
      backgroundColor: Colors.grey[50], // Light grey/off-white background

      body: SafeArea( // Keeps content within safe display areas (away from notches)
        child: Padding(
          // Add padding around the entire content column
          padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 16.0),
          child: Column(
            // Main axis alignment - can be adjusted (start, center, etc.)
            mainAxisAlignment: MainAxisAlignment.start,
            // Cross axis alignment - Align children horizontally
            crossAxisAlignment: CrossAxisAlignment.center, // Center logo and image initially
            children: <Widget>[
              // SizedBox for space from top (or use mainAxisAlignment)
              SizedBox(height: screenHeight * 0.05), // 5% of screen height space

              // 1. JibJob Logo
              Image.asset(
                'assets/images/jibjob_logo_full.png', // Ensure filename matches EXACTLY
                height: screenHeight * 0.06, // Adjust height as needed (e.g., 6% of screen height)
                // width: // You can also set width if needed
              ),

              // Space between logo and illustration
              SizedBox(height: screenHeight * 0.08), // Adjust spacing

              // 2. Main Illustration
              Image.asset(
                'assets/images/splash_illustration.png', // Ensure filename matches EXACTLY
                height: screenHeight * 0.35, // Adjust height (e.g., 35% of screen height)
                fit: BoxFit.contain, // Make sure image fits well
              ),

              // Space between illustration and text
              SizedBox(height: screenHeight * 0.1), // Adjust spacing

              // 3. Text Group (Left Aligned - Use a Column for this group)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start, // Left-align text within this group
                children: [
                  // Heading Text
                  Text(
                    'Trouvez Votre\nDream Job Ici!', // Using \n for line break
                    textAlign: TextAlign.start, // Explicitly align text start
                    style: TextStyle(
                      fontSize: 32, // Adjust font size
                      fontWeight: FontWeight.bold, // Make it bold
                      color: Colors.black87, // Adjust color if needed
                      height: 1.2, // Adjust line spacing if needed
                    ),
                  ),

                  // Space between heading and sub-text
                  SizedBox(height: 12),

                  // Sub-text
                  Text(
                    'Explorez tous les postes les plus passionnants en fonction de vos intérêts et de votre spécialisation.',
                    textAlign: TextAlign.start,
                    style: TextStyle(
                      fontSize: 16, // Adjust font size
                      color: Colors.grey[600], // Grey color
                      height: 1.4, // Adjust line spacing
                    ),
                  ),
                ],
              ),

              // Spacer pushes the button to the bottom
              const Spacer(),

              // 4. Arrow Button (Aligned to the bottom right)
              Align(
                alignment: Alignment.centerRight, // Align the button container to the right
                child: Container(
                  // Make the button circular and give it color
                  decoration: BoxDecoration(
                    color: const Color(0xFF483EA8), // Dark purple (adjust hex code from Figma)
                    shape: BoxShape.circle,
                  ),
                  // IconButton provides the ink splash effect on press
                  child: IconButton(
                    icon: const Icon(Icons.arrow_forward, color: Colors.white),
                    iconSize: 30, // Adjust icon size
                    onPressed: () {
                      print('Arrow button pressed!');
                      // TODO: Implement navigation to the next screen (e.g., LoginScreen)
                      Navigator.pushReplacement(
                        context,
                        MaterialPageRoute(builder: (context) => const LoginScreen()),
                      );
                    },
                  ),
                ),
              ),
              // Add some space below the button if needed
              SizedBox(height: screenHeight * 0.02),
            ],
          ),
        ),
      ),
    );
  }
}