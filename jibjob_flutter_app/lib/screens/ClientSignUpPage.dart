import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
// import 'Profileclient.dart'; // We'll comment this out for now if not immediately used for navigation AFTER signup
import 'Client3State.dart';

class ClientSignUpPage extends StatefulWidget {
  const ClientSignUpPage({Key? key}) : super(key: key);

  @override
  State<ClientSignUpPage> createState() => _ClientSignUpPageState();
}

class _ClientSignUpPageState extends State<ClientSignUpPage> {
  // --- State Variables
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _fullNameController = TextEditingController();
  final _phoneController = TextEditingController();
  final _cityController = TextEditingController();
  final _presentationController = TextEditingController();
  // We could add a controller for profile picture if we implement picking, but not for now.
  bool _isLoading = false; // To show a loading indicator

  // --- Supabase Client ---
  final _supabase = Supabase.instance.client; // Easy access to Supabase client (from your version)

  //colors for the app
  final Color darkPurple = Color(0xFF130160);
  final Color jibJobPurple = Color(0xFF130160);

  // --- Dispose controllers when widget is removed ---
  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    _fullNameController.dispose();
    _phoneController.dispose();
    _cityController.dispose();
    _presentationController.dispose();
    super.dispose();
  }

  // --- Sign Up Function
  Future<void> _signUp() async {
    if (!_formKey.currentState!.validate()) {
      // Lets use TextFormField instead of buildFields
      if (_emailController.text.trim().isEmpty || _passwordController.text.trim().isEmpty || _fullNameController.text.trim().isEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Please fill in Email, Password, and Full Name.')),
        );
        return;
      }
      if (_passwordController.text.trim().length < 6) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Password must be at least 6 characters.')),
        );
        return;
      }

    }


    setState(() {
      _isLoading = true;
    });

    final email = _emailController.text.trim();
    final password = _passwordController.text.trim();
    final fullName = _fullNameController.text.trim();
    final phoneNumber = _phoneController.text.trim();
    final locationText = _cityController.text.trim();
    final bio = _presentationController.text.trim();

    try {
      final AuthResponse authResponse = await _supabase.auth.signUp(
        email: email,
        password: password,
      );

      if (authResponse.user != null) {
        // Insert into public.users table
        await _supabase.from('users').insert({
          'auth_user_id': authResponse.user!.id,
          'email': email,
          'full_name': fullName,
          'phone_number': phoneNumber.isNotEmpty ? phoneNumber : null, // Save if provided
          'location_text': locationText.isNotEmpty ? locationText : null, // Save if provided
          'bio': bio.isNotEmpty ? bio : null, // Save if provided
          // profile_picture_url would be handled after image upload
        });

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Sign up successful!')),
          );
          print("Sign up success! TODO: Navigate to next page.");
        }
      }
    } on AuthException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Sign up failed: ${e.message}')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('An unexpected error occurred: ${e.toString()}')),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }


  // IMPORTANT CHANGE: Modified buildField to return TextFormField for validation.
  Widget buildField(String label, String hint, TextEditingController controller,
      {String? subLabel, bool isPassword = false, String? Function(String?)? validator, TextInputType? keyboardType, TextInputAction? textInputAction}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: TextStyle(fontSize: 18, color: jibJobPurple, fontWeight: FontWeight.w500)), // Added color and weight
          if (subLabel != null)
            Padding(
              padding: const EdgeInsets.only(top: 2.0, bottom: 4.0), // Added bottom padding
              child: Text(
                subLabel,
                style: TextStyle(fontSize: 12, color: Colors.black54),
              ),
            ),
          SizedBox(height: 6),
          TextFormField( // CHANGED from TextField to TextFormField
            controller: controller,
            obscureText: isPassword,
            decoration: InputDecoration(
              hintText: hint,
              border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8), // Standardized radius
                  borderSide: BorderSide(color: Colors.grey.shade400)
              ),
              focusedBorder: OutlineInputBorder( // Added focused border
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide(color: jibJobPurple, width: 2)
              ),
              enabledBorder: OutlineInputBorder( // Added enabled border
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide(color: Colors.grey.shade400)
              ),
              contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 14), // Adjusted padding
              fillColor: Colors.grey[100], // Lighter fill
              filled: true,
            ),
            validator: validator, // Added validator
            keyboardType: keyboardType,
            textInputAction: textInputAction,
          ),
        ],
      ),
    );
  }

  // Modified buildFieldLarge to use TextFormField
  Widget buildFieldLarge(String label, String hint, TextEditingController controller,
      {String? subLabel, String? Function(String?)? validator}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: TextStyle(fontSize: 18, color: jibJobPurple, fontWeight: FontWeight.w500)),
          if (subLabel != null)
            Padding(
              padding: const EdgeInsets.only(top: 2.0, bottom: 4.0),
              child: Text(
                subLabel,
                style: TextStyle(fontSize: 12, color: Colors.black54),
              ),
            ),
          SizedBox(height: 6),
          TextFormField( // CHANGED from TextField to TextFormField
            maxLines: 5,
            controller: controller,
            decoration: InputDecoration(
              hintText: hint,
              border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide(color: Colors.grey.shade400)
              ),
              focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide(color: jibJobPurple, width: 2)
              ),
              enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide(color: Colors.grey.shade400)
              ),
              contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 14),
              fillColor: Colors.grey[100],
              filled: true,
            ),
            validator: validator, // Added validator
            textInputAction: TextInputAction.newline, // Allow multiple lines
          ),
        ],
      ),
    );
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      // For a signup page, usually there's no bottom navigation.
      // If you want it, uncomment, but ensure navigation from it is handled.
      /*
      bottomNavigationBar: BottomNavigationBar(
        selectedItemColor: jibJobPurple,
        unselectedItemColor: Colors.grey,
        //currentIndex: 4, // This should be dynamic if used on multiple pages
        type: BottomNavigationBarType.fixed,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Accueil'),
          BottomNavigationBarItem(icon: Icon(Icons.business), label: 'Pros'),
          BottomNavigationBarItem(icon: Icon(Icons.add_circle, size: 40), label: ''),
          BottomNavigationBarItem(icon: Icon(Icons.notifications), label: 'Demandes'),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Compte'),
        ],
        onTap: (index) {
          // TODO: Handle navigation for bottom bar
          print("Bottom nav tapped: $index");
        },
      ),
      */

      // Using AppBar instead of custom container for consistency and standard features.
      appBar: AppBar(
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: jibJobPurple),
          onPressed: () {
            if (Navigator.canPop(context)) {
              Navigator.pop(context);
            } else {
              // Fallback if there's nothing to pop, e.g. direct navigation to Client3State
              Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => Client3State()));
            }
          },
        ),
        title: Text(
          'Creer un compte client',
          style: TextStyle(
            fontFamily: 'DM Sans', // Ensure this font is added to pubspec.yaml and assets
            fontSize: 22, // Slightly smaller for AppBar
            fontWeight: FontWeight.bold, // Bolder for title
            color: jibJobPurple,
          ),
        ),
        backgroundColor: Colors.grey[100], // Light background for AppBar
        elevation: 1.0, // Slight shadow
        centerTitle: false, // Align title to the start after leading icon
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding( // Added padding for the whole form content
            padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 20.0),
            child: Form( // Wrap Column with Form
              key: _formKey,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Removed the custom top bar, using AppBar now.
                  // SizedBox(height: 24), // Adjust spacing as AppBar is now standard

                  // Form fields using adapted buildField method
                  buildField(
                    "Nom et Prenom*",
                    "ex: Omar Laggoune",
                    _fullNameController,
                    validator: (value) {
                      if (value == null || value.isEmpty) return 'Veuillez entrer votre nom complet';
                      return null;
                    },
                    textInputAction: TextInputAction.next,
                  ),
                  buildField(
                    "Email*",
                    "ex: omar.laggoune@exemple.com",
                    _emailController,
                    keyboardType: TextInputType.emailAddress,
                    validator: (value) {
                      if (value == null || value.isEmpty) return 'Veuillez entrer votre email';
                      if (!RegExp(r"^[a-zA-Z0-9.]+@[a-zA-Z0-9]+\.[a-zA-Z]+").hasMatch(value)) return 'Veuillez entrer un email valide';
                      return null;
                    },
                    textInputAction: TextInputAction.next,
                  ),
                  buildField(
                    "Mot de passe*",
                    "Minimum 6 caractères",
                    _passwordController,
                    isPassword: true,
                    validator: (value) {
                      if (value == null || value.isEmpty) return 'Veuillez entrer un mot de passe';
                      if (value.length < 6) return 'Le mot de passe doit comporter au moins 6 caractères';
                      return null;
                    },
                    textInputAction: TextInputAction.next, // Changed to next, last field will be .done
                  ),
                  buildField(
                    "Telephone", // Optional
                    "ex: 06 12 34 56 78",
                    _phoneController,
                    keyboardType: TextInputType.phone,
                    subLabel: "Pour recevoir les demandes des clients (optionnel)",
                    textInputAction: TextInputAction.next,
                    // No validator making it optional
                  ),
                  buildField(
                    "Commune / Ville", // Optional
                    "ex: Birkhadem ou code postal",
                    _cityController,
                    subLabel: "Pour les propositions de services (optionnel)",
                    textInputAction: TextInputAction.next,
                    // No validator making it optional
                  ),
                  buildFieldLarge(
                    "Presentation", // Optional
                    "ex: Je suis disponible pour divers petits boulots...",
                    _presentationController,
                    subLabel: 'Presentez vous en quelques mots (optionnel)',
                    // No validator making it optional
                  ),
                  const SizedBox(height: 24),
                  // --- Photo de Profile (functionality TBD) ---
                  // This needs more work khali 3liha pour linstant (image_picker package, Supabase storage upload)
                  // For MVP, this can be a non-functional UI element or skipped.
                  Align( // Center the "Photo de Profile" section
                    alignment: Alignment.center,
                    child: Column(
                      children: [
                        Text('Photo de Profile (Optionnel)', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500, color: jibJobPurple)),
                        SizedBox(height: 8),
                        GestureDetector(
                          onTap: () {
                            // TODO: Implement image picking logic
                            print("Image picker tapped - TODO");
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(content: Text('Fonctionnalité de photo de profil à venir!')),
                            );
                          },
                          child: CircleAvatar(
                            radius: 50, // Increased radius
                            backgroundColor: Colors.grey[300],
                            child: Icon(Icons.camera_alt, size: 40, color: jibJobPurple.withOpacity(0.7)),
                          ),
                        )
                      ],
                    ),
                  ),
                  const SizedBox(height: 40), // Increased spacing
                  // --- Submit button ---
                  _isLoading
                      ? const Center(child: CircularProgressIndicator())
                      : ElevatedButton(
                    // onPressed: _signUp, // Call your signUp logic!
                    onPressed: () {
                      if (_formKey.currentState!.validate()) {
                        _signUp();
                      }
                    },
                    style: ElevatedButton.styleFrom(
                        backgroundColor: jibJobPurple, // Using defined color
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        minimumSize: Size(double.infinity, 50),
                        padding: const EdgeInsets.symmetric(vertical: 16) // Added for consistency
                    ),
                    child: Text(
                        "Creer mon compte", // Changed text from "Suivant"
                        style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w500) // Adjusted style
                    ),
                  ),
                  const SizedBox(height: 16.0),
                  // --- Option to go to Login Page (From your original clean code) ---
                  Align(
                    alignment: Alignment.center,
                    child: TextButton(
                      onPressed: () {
                        // TODO: Navigate to Login Page properly
                        // Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (context) => LoginPage()));
                        print("Navigate to Login Page - TODO");
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('Navigation vers la page de connexion à implémenter.')),
                        );
                      },
                      child: Text(
                        'Vous avez déjà un compte? Se connecter',
                        style: TextStyle(color: jibJobPurple, fontWeight: FontWeight.w500),
                      ),
                    ),
                  )
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}