import 'dart:io';
import 'package:flutter/material.dart';
import 'Profileclient.dart';
import 'Client3State.dart';
import 'package:image_picker/image_picker.dart';


class ClientSignUpPage extends StatefulWidget {
  @override
  _ClientSignUpPageState createState() => _ClientSignUpPageState();
}

// ignore: unused_element
class _ClientSignUpPageState extends State<ClientSignUpPage> {

  final ImagePicker _picker = ImagePicker();

  final emailController = TextEditingController();
  final passwordController = TextEditingController();
  final nameController = TextEditingController();
  final phoneController = TextEditingController();
  final cityController = TextEditingController();
  final presentationController = TextEditingController();
  XFile? imageController ;  

  XFile? _image; // To store the selected image

  // Function to pick an image
  Future<void> _pickImage() async {
    try {
      // Pick an image from the gallery
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);

      if (image != null) {
        setState(() {
          _image = image;
          imageController = _image ;
        });
      } else {
        // Handle case where user cancels the picker
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('No image selected.')),
        );
      }
    } catch (e) {
      // Handle any errors that occur
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to pick image: $e')),
      );
    }
  }
  
  final Color darkPurple = Color(0xFF20004E);

  @override
  void dispose() {
    emailController.dispose();
    passwordController.dispose();
    nameController.dispose();
    phoneController.dispose();
    cityController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
          double screenHeight = MediaQuery.of(context).size.height;

    return Scaffold(
      bottomNavigationBar: BottomNavigationBar(
        selectedItemColor: darkPurple,
        unselectedItemColor: Colors.grey,
        currentIndex: 4,
        type: BottomNavigationBarType.fixed,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Accueil'),
          BottomNavigationBarItem(icon: Icon(Icons.business), label: 'Pros'),
          BottomNavigationBarItem(icon: Icon(Icons.add_circle, size: 40), label: ''),
          BottomNavigationBarItem(icon: Icon(Icons.notifications), label: 'Demandes'),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Compte'),
        ],
      ),

      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start ,
            children: [
              // Top bar
              Container ( color: Colors.grey[200] ,
                  padding: EdgeInsets.symmetric(horizontal: 10 , vertical: 16 ) ,
                  child : Row(
                    children: [
                      IconButton(
                        icon: Icon(Icons.arrow_back, color: darkPurple),
                        iconSize: 35,
                        onPressed: () {
                              Navigator.push(context, MaterialPageRoute(builder: (_) => Client3State()));
                        },
                      ),
                      Text(
                        'Creer un compte client',
                        style: TextStyle(
                          fontFamily: 'DM Sans' ,
                          fontSize: 18,
                          fontWeight: FontWeight.w400,
                          color: Color(0xFF130160),
                        ),
                      ),
                    ],
                  )),
              SizedBox(height: 24),

              // Form fields
              Container( padding: EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                  child : Column( crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        buildField("Email", "Ex: zakarya@gmail.com", emailController),
                        buildField("Mot de pass", "********", passwordController, isPassword: true),
                        buildField("Nom et Prenom", "ex: Zakarya Oukil", nameController),
                        buildField("Telephone", "ex:  68 71 87 84", phoneController,
                            subLabel: "Pour recevoir les demandes des clients"),
                        buildField("Commune", "ex: Birkhadem ou code postal", cityController,
                            subLabel: "Il est preferable de rechercher par code postal"),

                        buildFieldLarge(
                            "Presentation", "ex: jai x annees d’experience en x. Le peux fournir le service a, service b, service c. Je suis joignable de x jour a x jour, de x heure a x heure." ,
                            presentationController ,
                            subLabel:'Presenter vous en queleques mots aux clients'
                        ) ,


                        SizedBox(height: 24),

                        Column(
                          children: [
                            SizedBox(height: screenHeight * 0.03),
                            Container(
                              height: screenHeight * 0.2,
                              width: screenHeight * 0.2,
                              decoration: BoxDecoration(
                                color: Colors.grey[350] ,
                                borderRadius: BorderRadius.circular(1000),
                                border: Border.all(
                                  color: Color(0xFF130160),
                                  width: 2,
                                ), 
                              ),

                              child: GestureDetector(
                                onTap: _pickImage,
                                child: _image == null
                                    ? ClipOval(
                                      child: Image.asset(
                                        "assets/Picture_Icon.png",
                                        fit: BoxFit.cover,
                                        width: double.infinity,
                                        height: double.infinity,
                                      ))
                                    : ClipOval(
                                      child: Image.file(
                                        File(_image!.path),
                                        fit: BoxFit.cover,
                                        width: double.infinity,
                                        height: double.infinity,
                                      )),
                              ),
                            
                            )
                          ],
                        ),

                        SizedBox(height: 50,) ,

                        // Submit button
                        ElevatedButton(
                          onPressed: () {
                                Navigator.push(context, MaterialPageRoute(builder: (_) => Profileclient(emailController , passwordController , nameController , phoneController , cityController , presentationController , imageController)));
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Color(0xFF130160),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            minimumSize: Size(double.infinity, 50),
                          ),
                          child: Text("Suivant" , style : TextStyle(color: Colors.white , fontSize: 25 , fontWeight: FontWeight.w400)),
                        )])),
            ],
          ),
        ),
      ),
    );
  }

  Widget buildField(String label, String hint, TextEditingController controller,
      {String? subLabel, bool isPassword = false}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label , style: TextStyle(fontSize: 18)),
          if (subLabel != null)
            Padding(
              padding: const EdgeInsets.only(top: 2.0),
              child: Text(
                subLabel,
                style: TextStyle(fontSize: 12, color: Colors.black54),
              ),
            ),
          SizedBox(height: 6),
          TextField(
            controller: controller,
            obscureText: isPassword,
            decoration: InputDecoration(
              hintText: hint,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              contentPadding: EdgeInsets.symmetric(horizontal: 12),
              fillColor: Colors.grey[200] ,
              filled: true ,
            ),
          ),
        ],
      ),
    );
  }


    Widget buildFieldLarge(String label, String hint, TextEditingController controller,
        {String? subLabel, bool isPassword = false}) {
      return Padding(
        padding: const EdgeInsets.only(bottom: 16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label , style: TextStyle(fontSize: 18)),
            if (subLabel != null)
              Padding(
                padding: const EdgeInsets.only(top: 2.0),
                child: Text(
                  subLabel,
                  style: TextStyle(fontSize: 12, color: Colors.black54),
                ),
              ),
            SizedBox(height: 6),
            TextField(
              maxLines: 5,
              controller: controller,
              decoration: InputDecoration(
                hintText: hint,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                contentPadding: EdgeInsets.symmetric(horizontal: 12),
                fillColor: Colors.grey[200] ,
                filled: true ,
              ),
            ),
          ],
        ),
      );
    }
}