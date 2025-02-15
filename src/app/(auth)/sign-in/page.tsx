import { SignInForm } from "@/src/components/sign/sign-in-form";
import Image from 'next/image';

export default function SignIn() {

  return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-6 bg-cover bg-center bg-no-repeat p-6 md:p-10">
        <img
          src="/dashboard.jpeg"
          alt="Dashboard Background"
          width={1702}
          height={989}
          className="absolute top-0 left-0 -z-10 h-full w-full object-cover filter blur-sm"
        />
        <div className="flex w-full max-w-sm flex-col gap-6">
          <SignInForm />
        </div>
      </div>
  );
}
