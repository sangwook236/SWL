#!/usr/bin/env python

import random
from captcha.image import ImageCaptcha
from captcha.audio import AudioCaptcha
import matplotlib.pyplot as plt

def create_random_text(captcha_string_size=10, is_digit=False):
	alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
	digit = '0123456789'
	charset = digit if is_digit else alphabet + digit

	return ''.join(random.choices(charset, k=captcha_string_size))

# REF [site] >> https://www.dev2qa.com/how-to-generate-random-captcha-in-python/
def create_image_captcha(captcha_text):
	image_captcha = ImageCaptcha(width=160, height=60, fonts=None, font_sizes=None)

	# Create the captcha image.
	image = image_captcha.generate_image(captcha_text)

	# Add noise curve for the image.
	image_captcha.create_noise_curve(image, image.getcolors())

	# Add noise dots for the image.
	image_captcha.create_noise_dots(image, image.getcolors())

	# Save the image to a png file.
	image_filepath = './captcha.png'
	image_captcha.write(captcha_text, image_filepath)

	# Display the image in a matplotlib viewer.
	plt.imshow(image)
	plt.show()

	print(image_filepath + ' has been created.')

# REF [site] >> https://www.dev2qa.com/how-to-generate-random-captcha-in-python/
def create_audio_captcha(captcha_text=None):
	# Create the audio captcha with the specified voice wav file library folder.
	# Each captcha char should have its own directory under the specified folder (such as ./voices),
	# for example ./voices/a/a.wav will be played when the character is a.
	# If you do not specify your own voice file library folder, the default built-in voice library which has only digital voice file will be used. 
	# audio_captcha = AudioCaptcha(voicedir='./voices')

	# Create an audio captcha which use digital voice file only.
	audio_captcha = AudioCaptcha()

	if captcha_text is None:
		captcha_text = audio_captcha.random(length=random.randint(5, 10))

	# Generate the audio captcha file.
	audio_data = audio_captcha.generate(captcha_text)

	# Add background noise for the audio.
	#audio_captcha.create_background_noise(random.randint(5, 10), audio_captcha.random(length=random.randint(5, 10)))

	# Save the autiod captcha file.
	audio_filepath = './captcha.wav'
	audio_captcha.write(captcha_text, audio_filepath)

	print(audio_filepath + ' has been created.')

def main():
	# Create image captcha.
	text = create_random_text(5, is_digit=False)
	create_image_captcha(text)

	# Create audio captcha.
	text = create_random_text(5, is_digit=True)
	create_audio_captcha(text)

#--------------------------------------------------------------------

# Usage:
#	pip install captcha

if '__main__' == __name__:
	main()
